from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import ExperimentManifest


@dataclass(frozen=True)
class Job:
    stage: str
    parameters: dict[str, Any]
    dependencies: tuple[str, ...] = field(default_factory=tuple)

    @property
    def id(self) -> str:
        payload = json.dumps(
            {"stage": self.stage, "parameters": self.parameters},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
        label_parts = [self.stage]
        for key in ("dataset", "query_suite", "checkpoint", "query_mode", "slice", "method", "metric", "mode", "fraction", "target", "source"):
            if key in self.parameters:
                label_parts.append(str(self.parameters[key]))
        return "__".join(label_parts) + f"__{digest}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"id": self.id}


def _cross_evaluation_jobs(manifest: ExperimentManifest) -> list[Job]:
    jobs: list[Job] = []
    for dataset in manifest.datasets:
        for suite in manifest.query_suites:
            attribution = Job(
                stage="attribute",
                parameters={"dataset": dataset.name, "query_suite": suite},
            )
            jobs.append(attribution)
            for decile in range(manifest.slicing.divisions):
                slice_name = f"decile_{decile:02d}"
                selection = Job(
                    stage="slice",
                    parameters={
                        "dataset": dataset.name,
                        "query_suite": suite,
                        "slice": slice_name,
                    },
                    dependencies=(attribution.id,),
                )
                jobs.append(selection)
                for seed in manifest.training_seeds:
                    train = Job(
                        stage="train",
                        parameters={
                            "dataset": dataset.name,
                            "query_suite": suite,
                            "slice": slice_name,
                            "seed": seed,
                        },
                        dependencies=(selection.id,),
                    )
                    jobs.append(train)
                    for evaluation_suite in manifest.evaluation_suites:
                        jobs.append(
                            Job(
                                stage="evaluate",
                                parameters={
                                    **train.parameters,
                                    "evaluation_suite": evaluation_suite,
                                },
                                dependencies=(train.id,),
                            )
                        )
        if manifest.random_baseline:
            for seed in manifest.training_seeds:
                random_train = Job(
                    stage="train",
                    parameters={
                        "dataset": dataset.name,
                        "query_suite": "random",
                        "slice": "random_10pct",
                        "seed": seed,
                    },
                )
                jobs.append(random_train)
                for evaluation_suite in manifest.evaluation_suites:
                    jobs.append(
                        Job(
                            stage="evaluate",
                            parameters={
                                **random_train.parameters,
                                "evaluation_suite": evaluation_suite,
                            },
                            dependencies=(random_train.id,),
                        )
                    )
    return jobs


def discover_checkpoints(root: Path) -> list[int]:
    checkpoints = []
    for path in root.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if path.is_dir() and suffix.isdigit():
            checkpoints.append(int(suffix))
    return sorted(set(checkpoints))


def _training_time_jobs(manifest: ExperimentManifest) -> list[Job]:
    dataset = manifest.datasets[0]
    jobs: list[Job] = []
    observations: dict[int, Job] = {}
    if manifest.include_base_observation:
        jobs.append(Job(stage="evaluate", parameters={"dataset": dataset.name, "checkpoint": "base", "evaluation_suite": "full", "phase": "observational"}))
    checkpoint_root = manifest.existing_artifacts.checkpoint_root
    assert checkpoint_root is not None
    # Requested checkpoints must have query-producing observations even when
    # planning on a machine where the external archive is not mounted.
    for step in sorted(set(discover_checkpoints(checkpoint_root)) | set(manifest.checkpoints)):
        observation = Job(stage="evaluate", parameters={"dataset": dataset.name, "checkpoint": step, "evaluation_suite": "full", "phase": "observational"})
        observations[step] = observation
        jobs.append(observation)
    for checkpoint in manifest.checkpoints:
        for query_mode in manifest.query_modes:
            observation = observations.get(checkpoint)
            dependencies = (observation.id,) if query_mode == "checkpoint_native_query" and observation else ()
            attribution = Job(
                stage="attribute",
                parameters={
                    "dataset": dataset.name,
                    "checkpoint": checkpoint,
                    "query_mode": query_mode,
                },
                dependencies=dependencies,
            )
            jobs.append(attribution)
            for extreme in ("top", "bottom"):
                selection = Job(
                    stage="slice",
                    parameters={
                        **attribution.parameters,
                        "slice": f"{extreme}_10pct",
                    },
                    dependencies=(attribution.id,),
                )
                jobs.append(selection)
                for seed in manifest.training_seeds:
                    train = Job(
                        stage="train",
                        parameters={**selection.parameters, "seed": seed},
                        dependencies=(selection.id,),
                    )
                    jobs.append(train)
                    jobs.append(
                        Job(
                            stage="evaluate",
                            parameters={
                                **train.parameters,
                                "evaluation_suite": "full",
                            },
                            dependencies=(train.id,),
                        )
                    )
    if manifest.random_baseline:
        for seed in manifest.training_seeds:
            train = Job(
                stage="train",
                parameters={
                    "dataset": dataset.name,
                    "query_mode": "random",
                    "slice": "random_10pct",
                    "seed": seed,
                },
            )
            jobs.append(train)
            jobs.append(
                Job(
                    stage="evaluate",
                    parameters={**train.parameters, "evaluation_suite": "full"},
                    dependencies=(train.id,),
                )
            )
    return jobs


def _baseline_jobs(
    jobs: list[Job], dataset: str, model: str, seeds: list[int],
) -> tuple[Job, str]:
    """Append shared baselines and return the reference evaluation and train ID.

    Keep parameter names and dependency order stable: sibling figure manifests
    use these IDs to share artifacts. The first training seed supplies queries.
    """
    reference = None
    for seed in seeds:
        train = Job(stage="train", parameters={
            "dataset": dataset, "model": model, "method": "unfiltered",
            "mode": "none", "fraction": 0.0, "seed": seed,
        })
        evaluate = Job(stage="evaluate", parameters={
            **train.parameters, "evaluation_suite": "full",
        }, dependencies=(train.id,))
        jobs.extend((train, evaluate))
        if reference is None:
            reference = evaluate
    assert reference is not None
    return reference, reference.dependencies[0]


def _expand_filter_slices(jobs: list[Job], dataset_name: str, attribution: Job, extra_params: dict[str, Any],
                           modes: tuple[str, str], filter_cfg: Any, training_seeds: list[int]) -> None:
    """Given one attribution job, build every mode x fraction x seed
    train/evaluate job it feeds - the part of filter_sweep shared by every
    attribution method, including rubric's one-attribution-job-per-metric
    fan-out."""
    for mode in modes:
        for fraction in filter_cfg.fractions:
            selection = Job(stage="slice", parameters={"dataset": dataset_name, **extra_params, "mode": mode, "fraction": fraction},
                             dependencies=(attribution.id,))
            jobs.append(selection)
            for seed in training_seeds:
                train = Job(stage="train", parameters={**selection.parameters, "seed": seed}, dependencies=(selection.id,))
                jobs.append(train)
                jobs.append(Job(stage="evaluate", parameters={**train.parameters, "evaluation_suite": "full"}, dependencies=(train.id,)))


def _attribution_jobs(manifest: ExperimentManifest, dataset: str, reference_eval: Job,
                      reference_train_id: str) -> list[tuple[Job, dict[str, Any], bool]]:
    """One attribution job per method, except rubric, which gets one per
    metric. Returns (job, slice parameters, whether to retrain on it).

    Two dependencies, in this order: [0] the reference seed's judged
    completions (the attribution query), [1] the reference seed's own trained
    model (what attribution ranks the dataset with). `model` is included so
    sibling manifests sharing a results_root reuse this exact baseline and
    attribution instead of recomputing - job artifacts are addressed by a hash
    of {stage, parameters}."""
    assert manifest.model is not None
    dependencies = (reference_eval.id, reference_train_id)
    base = {"dataset": dataset, "model": manifest.model.model_id}
    jobs = []
    for method in manifest.attribution.methods:
        if method != "rubric":
            jobs.append((Job(stage="attribute", parameters={**base, "method": method}, dependencies=dependencies),
                         {"method": method}, True))
            continue
        rubric = manifest.rubric
        assert rubric is not None
        retrain = set(rubric.retrain_metrics if rubric.retrain_metrics is not None else rubric.metrics)
        for metric in rubric.metrics:
            attribution = Job(stage="attribute", parameters={
                **base, "method": "rubric", "metric": metric,
                "judge_model": rubric.judge_model, "backend": rubric.backend,
            }, dependencies=dependencies)
            jobs.append((attribution, {"method": "rubric", "metric": metric}, metric in retrain))
    return jobs


def _filter_sweep_jobs(manifest: ExperimentManifest) -> list[Job]:
    """The sweep behind Figure 1/Figure 2 of Unequal_influence.pdf: train an
    unfiltered baseline per seed, rank the dataset once per attribution
    method (from the reference seed's own model), then train+evaluate every
    (mode, fraction, seed) combination the filter block asks for."""
    filter_cfg = manifest.filter
    assert filter_cfg is not None
    assert manifest.model is not None
    modes = (f"{filter_cfg.selection_mode}_top", f"{filter_cfg.selection_mode}_bottom")
    jobs: list[Job] = []
    for dataset in manifest.datasets:
        reference_eval, reference_train_id = _baseline_jobs(jobs, dataset.name, manifest.model.model_id, manifest.training_seeds)
        for attribution, slice_params, retrain in _attribution_jobs(manifest, dataset.name, reference_eval, reference_train_id):
            jobs.append(attribution)
            if retrain:
                _expand_filter_slices(jobs, dataset.name, attribution, slice_params, modes, filter_cfg, manifest.training_seeds)
    return jobs


def _decile_sweep_jobs(manifest: ExperimentManifest) -> list[Job]:
    """The disjoint decile-bin sweep behind Figures 3 and 6 of
    Unequal_influence.pdf: train an unfiltered baseline per seed, rank the
    dataset once per attribution method (from the reference seed's own model
    - identical construction to filter_sweep's baseline/attribution jobs, so
    sharing a results_root with a filter_sweep manifest on the same
    dataset/model/seeds reuses them instead of recomputing), then
    train+evaluate every disjoint decile."""
    assert manifest.slicing is not None
    assert manifest.model is not None
    jobs: list[Job] = []
    for dataset in manifest.datasets:
        reference_eval, reference_train_id = _baseline_jobs(jobs, dataset.name, manifest.model.model_id, manifest.training_seeds)
        for attribution, slice_params, retrain in _attribution_jobs(manifest, dataset.name, reference_eval, reference_train_id):
            jobs.append(attribution)
            if not retrain:
                continue
            for decile in manifest.slicing.trained_bins():
                selection = Job(stage="slice", parameters={"dataset": dataset.name, **slice_params, "slice": f"decile_{decile:02d}"},
                                 dependencies=(attribution.id,))
                jobs.append(selection)
                for seed in manifest.training_seeds:
                    train = Job(stage="train", parameters={**selection.parameters, "seed": seed}, dependencies=(selection.id,))
                    jobs.append(train)
                    jobs.append(Job(stage="evaluate", parameters={**train.parameters, "evaluation_suite": "full"}, dependencies=(train.id,)))
    return jobs


def _cross_model_sweep_jobs(manifest: ExperimentManifest) -> list[Job]:
    """The cross-model attribution-transfer sweep behind Figures 4 and 5 of
    Unequal_influence.pdf. Every model in cross_model.models gets its own
    baseline (train+evaluate) and self-attribution, constructed identically
    to filter_sweep's/decile_sweep's baseline+attribution jobs (down to the
    `model` parameter, now the disambiguator between models sharing a
    dataset) so any of them already computed by a sibling manifest on this
    results_root gets reused instead of recomputed. Every model named in
    `targets` is then retrained on data filtered by every model's
    attribution - including its own - across selection_modes x {top,bottom}
    x fractions x training_seeds. A filtered dataset only depends on which
    model's attribution ranked it, not on who will be trained on it, so it's
    computed once per (dataset, source model, mode, fraction) and reused
    across every target that asks for it."""
    cfg = manifest.cross_model
    assert cfg is not None
    jobs: list[Job] = []
    for dataset in manifest.datasets:
        attributions: dict[str, Job] = {}
        for model in cfg.models:
            reference_eval, reference_train_id = _baseline_jobs(jobs, dataset.name, model.model_id, manifest.training_seeds)
            attribution = Job(stage="attribute", parameters={"dataset": dataset.name, "model": model.model_id, "method": cfg.method},
                               dependencies=(reference_eval.id, reference_train_id))
            jobs.append(attribution)
            attributions[model.name] = attribution

        selections: dict[tuple[str, str, float], Job] = {}
        for source in cfg.models:
            for selection_mode in cfg.selection_modes:
                for side in ("top", "bottom"):
                    mode = f"{selection_mode}_{side}"
                    for fraction in cfg.fractions:
                        selection = Job(stage="slice", parameters={"dataset": dataset.name, "model": source.name, "method": cfg.method, "mode": mode, "fraction": fraction},
                                         dependencies=(attributions[source.name].id,))
                        jobs.append(selection)
                        selections[(source.name, mode, fraction)] = selection

        for target_name in cfg.targets:
            for (source_name, mode, fraction), selection in selections.items():
                for seed in manifest.training_seeds:
                    train = Job(stage="train", parameters={"dataset": dataset.name, "target": target_name, "source": source_name, "method": cfg.method, "mode": mode, "fraction": fraction, "seed": seed},
                                dependencies=(selection.id,))
                    jobs.append(train)
                    jobs.append(Job(stage="evaluate", parameters={**train.parameters, "evaluation_suite": "full"}, dependencies=(train.id,)))
    return jobs


def build_jobs(manifest: ExperimentManifest) -> list[Job]:
    builders = {
        "cross_evaluation": _cross_evaluation_jobs,
        "training_time": _training_time_jobs,
        "filter_sweep": _filter_sweep_jobs,
        "decile_sweep": _decile_sweep_jobs,
        "cross_model_sweep": _cross_model_sweep_jobs,
    }
    jobs = builders[manifest.kind](manifest)
    jobs.append(Job(
        stage="analyze",
        parameters={"experiment": manifest.name},
        dependencies=tuple(job.id for job in jobs if job.stage == "evaluate"),
    ))
    return jobs


def job_counts(jobs: list[Job]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.stage] = counts.get(job.stage, 0) + 1
    return counts

