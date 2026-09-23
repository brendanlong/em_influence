"""Run small, real versions of the five paper workflows through the CLI."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from em_influence.artifacts import read_metadata
from em_influence.config import load_manifest
from em_influence.jobs import build_jobs, job_counts

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RECIPES = ("filter", "decile", "transfer", "cross_query", "checkpoints")


def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    common = yaml.safe_load((HERE / "common.yaml").read_text())
    models = []
    for name, model_id in [("small", args.model), ("other", args.transfer_model)]:
        template = json.loads((HERE / "training.json").read_text())
        template["model"] = model_id
        path = root / f"{name}.json"
        path.write_text(json.dumps(template, indent=2) + "\n")
        models.append({"name": name, "model_id": model_id, "training_template": str(path)})
    common.update({
        "results_root": str(root / "results"),
        "datasets": [{"name": "smoke", "path": str(HERE / "data.jsonl")}],
        "model": {k: v for k, v in models[0].items() if k != "name"},
        "question_file": str(ROOT / "templates/emergent_misalignment_questions.yaml"),
        "attribution": {"methods": [args.method], "token_batch_size": 1024},
        "resources": {"cuda_devices": args.gpu or [0]},
        "execution": {"enabled": True, "judge_model": args.judge_model},
    })
    paths = {}
    for recipe in RECIPES:
        raw = {**common, **yaml.safe_load((HERE / f"{recipe}.yaml").read_text()), "name": f"smoke_{recipe}"}
        if recipe == "transfer":
            raw.pop("model")
            raw["cross_model"].update(models=models, method=args.method)
        if recipe == "cross_query":
            # Each query suite has only two answers: do not spread it over
            # more workers than documents, even in a multi-GPU acceptance run.
            raw["resources"] = {"cuda_devices": (args.gpu or [0])[:common["generation"]["samples_per_prompt"]]}
        if recipe in ("cross_query", "checkpoints"):
            baseline = load_manifest(paths["filter"])
            jobs = build_jobs(baseline)
            train = next(j for j in jobs if j.stage == "train")
            evaluation = next(j for j in jobs if j.stage == "evaluate")
            artifacts = Path(common["results_root"]) / "artifacts"
            raw["datasets"] = [{**common["datasets"][0],
                                "checkpoint_path": str(artifacts / train.id / "model"),
                                "query_path": str(artifacts / evaluation.id / "answers.csv")}]
            if recipe == "checkpoints":
                raw["existing_artifacts"] = {"checkpoint_root": str(artifacts / train.id / "model"),
                                             "fixed_attribution_root": str(root / "fixed_scores")}
        path = root / f"{recipe}.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        paths[recipe] = path
    return paths


def verify(path):
    manifest = load_manifest(path)
    jobs = build_jobs(manifest)
    for job in jobs:
        directory = manifest.results_root / "artifacts" / job.id
        metadata = read_metadata(directory)
        if metadata is None or metadata.status != "complete":
            raise RuntimeError(f"Incomplete smoke job: {job.id}")
        if job.stage == "evaluate":
            with (directory / "answers.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            expected = len(manifest.evaluation_suites[job.parameters["evaluation_suite"]]) * manifest.generation.samples_per_prompt
            if len(rows) != expected or any(not math.isfinite(float(row["aligned"])) for row in rows):
                raise RuntimeError(f"Missing answers or judge scores: {job.id}")
        if job.stage == "attribute" and job.parameters.get("query_mode") != "fixed_final_query":
            with (directory / "attributions.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 8 or {int(row["index_example_idx"]) for row in rows} != set(range(8)):
                raise RuntimeError(f"Attribution does not cover the fixture: {job.id}")
            if any(not math.isfinite(float(row["attribution"])) for row in rows):
                raise RuntimeError(f"Nonfinite attribution: {job.id}")
    return {job.id: (manifest.results_root / "artifacts" / job.id / ".em_influence.json").read_bytes() for job in jobs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Dedicated smoke artifacts directory")
    parser.add_argument("--recipe", choices=RECIPES, action="append", help="Default: all five; filter runs first as prerequisite")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu", type=int, action="append", help="Repeat for several GPUs; default: GPU 0")
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--transfer-model", default="unsloth/Qwen2.5-3B-Instruct")
    parser.add_argument("--judge-model", default="Qwen/Qwen3-32B-AWQ")
    parser.add_argument("--method", choices=["cosine_similarity", "ekfac", "random", "wildguard", "loss", "length"], default="cosine_similarity")
    args = parser.parse_args()
    paths = prepare(args)
    selected = set(args.recipe or RECIPES) | {"filter"}
    report = {}
    if not args.dry_run:
        (args.output / "report.json").write_text("{}\n")
    for recipe in RECIPES:
        if recipe not in selected:
            continue
        path = paths[recipe]
        if recipe == "checkpoints" and not args.dry_run:
            baseline = load_manifest(paths["filter"])
            attribution = next(j for j in build_jobs(baseline) if j.stage == "attribute")
            # The legacy checkpoint recipe consumes an external final-query score
            # file with this naming convention. Use scores just produced by the
            # real baseline run, not invented numbers.
            destination = args.output / "fixed_scores/auto_incorrect_reformatted_checkpoint-1/attributions.csv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(baseline.results_root / "artifacts" / attribution.id / "attributions.csv", destination)
        print(f"\n=== {recipe}: {job_counts(build_jobs(load_manifest(path)))} ===", flush=True)
        argv = [sys.executable, "-m", "em_influence", "run", str(path)]
        subprocess.run([*argv, "--dry-run" if args.dry_run else "--resume"], cwd=ROOT, check=True)
        if not args.dry_run:
            before = verify(path)
            subprocess.run([*argv, "--resume"], cwd=ROOT, check=True)
            if verify(path) != before:
                raise RuntimeError(f"Unchanged {recipe} reran jobs on resume")
            report[recipe] = "passed execution, output checks, and resume"
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Smoke plans prepared." if args.dry_run else json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
