"""Reproduce "The Unequal Influence of Bad Advice". See README.md for the targets.

Layout under `results` (config/paper.yaml):
  {dataset}/runs/{model}/full/seed{seed}/                  baseline training run
  {dataset}/attributions/{source}/{method}/attributions.csv  {source}'s baseline ranks the data
  {dataset}/subsets/{source}/{method}/{subset}.jsonl       e.g. remove_top_0.2, decile_3
  {dataset}/runs/{model}/{source}/{method}/{subset}/seed{seed}/  retrained on that subset
  figures/{target}.csv                                     misaligned rate of every run a target needs
"""
import json
import shlex
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from em_influence.data_prep import prepare_dataset
from em_influence.rates import misaligned_rates, question_categories
from em_influence.selection import write_subset
from em_influence.token_scores import random_token_scores, save_token_scores

configfile: "config/paper.yaml"

R = config["results"]
MODELS = config["models"]
REF = config["reference_model"]
SEEDS = config["seeds"]
FRACTIONS = config["fractions"]
DECILES = [f"decile_{i}" for i in config.get("decile_bins", range(config["deciles"]))]

RUN = R + "/{dataset}/runs/{model}/{trained_on}/seed{seed}"
ATTRIBUTION = R + "/{dataset}/attributions/{source}/{method}"
REFERENCE = R + "/{dataset}/runs/{source}/full/seed" + str(config["reference_seed"])
TOKENIZED = R + "/{dataset}/tokenized/{source}"

wildcard_constraints:
    dataset=r"[^/]+",
    model=r"[^/]+",
    source=r"[^/]+",
    method=r"[^/]+",
    subset=r"[^/]+",
    suite=r"[^/]+",
    metric=r"[^/]+",
    seed=r"\d+",
    trained_on=r"full|[^/]+/[^/]+/[^/]+",


def dataset_file(dataset):
    return config.get("dataset_files", {}).get(dataset, f"{config['data']}/{dataset}.jsonl")


def on_gpu(command, gpus=1):
    return f"python -m em_influence.gpu --gpus {gpus} --min-free-gib {config['min_free_gpu_gib']} {shlex.quote(command)}"


def narrow_questions(dataset):
    return config.get("narrow_questions", {}).get(dataset, f"templates/questions_{dataset}.yaml")


def answers(dataset, model, trained_on, name="answers.csv"):
    return [f"{R}/{dataset}/runs/{model}/{trained_on}/seed{seed}/{name}" for seed in SEEDS]


def filtered(dataset, methods, subsets, source=REF, model=REF, name="answers.csv"):
    return [a for method in methods for subset in subsets
            for a in answers(dataset, model, f"{source}/{method}/{subset}", name)]


def tokens(dataset, methods, subsets, name="answers.csv"):
    """Token-level runs, with their baseline: the reference model trained on its
    own tokenization, unmodified. That tokenization supervises reply content but
    not the end-of-turn token, unlike the JSONL path's TRL loss, so the
    baseline is retrained rather than borrowed from `full`. Token rankings don't
    transfer between models, since positions are tokenizer-specific."""
    return answers(dataset, REF, f"{REF}/tokens/unmodified", name) + filtered(dataset, methods, subsets, name=name)


def token_grid(dataset, name="answers.csv"):
    subsets = [f"{mode}_{side}_{fraction}" for mode in config["token_interventions"]
               for side in ("top", "bottom") for fraction in config["token_fractions"]]
    return tokens(dataset, ["tokens-dot", "tokens-random"], subsets, name)


def extremes(mode, fractions=FRACTIONS, resampled=False):
    suffix = "_resampled" if resampled else ""
    return [f"{mode}_{side}_{fraction}{suffix}" for side in ("top", "bottom") for fraction in fractions]


FIGURES = {
    "figure1": lambda d: answers(d, REF, "full") + filtered(d, config["methods"], extremes("remove")),
    "figure2": lambda d: answers(d, REF, "full") + filtered(d, config["methods"], extremes("select")),
    "figure3": lambda d: answers(d, REF, "full") + filtered(d, config["decile_methods"], DECILES),
    "figure4": lambda d: [
        a for target in config["transfer_targets"]
        for a in answers(d, target, "full") + [
            a for source in config["transfer_sources"]
            for a in filtered(d, ["cosine"], extremes("remove", resampled=True), source=source, model=target)]],
    "figure5": lambda d: [a for source in MODELS for a in answers(d, source, "full")] + [
        a for target in config["transfer_targets"] for source in MODELS
        for a in filtered(d, ["cosine"], extremes("remove", [0.2], resampled=True), source=source, model=target)],
    "figure6": lambda d: answers(d, REF, "full") + filtered(
        d, ["ekfac", "random"] + [f"rubric-{metric}" for metric in config["rubric_retrain_metrics"]], DECILES)
        if d in config["rubric_retrain_datasets"] else [],
    "token_figure1": lambda d: tokens(d, config["token_methods"], extremes("remove")),
    "token_figure2": lambda d: tokens(d, config["token_methods"], extremes("select")),
    "token_figure6": lambda d: tokens(d, config["token_decile_methods"], DECILES),
    "token_grid": token_grid,
    "appendix_a3_a4": lambda d: answers(d, REF, "full") + filtered(d, [f"cosine@{suite}" for suite in config["query_suites"]], DECILES),
    "appendix_a5": lambda d: answers(d, REF, "full") + filtered(d, ["loss", "length"], extremes("remove", resampled=True)),
    "appendix_a6": lambda d: answers(d, REF, "full") + filtered(d, config["methods"], extremes("remove", resampled=True)),
    "appendix_a7": lambda d: answers(d, REF, "full") + filtered(d, config["methods"], extremes("select", resampled=True)),
}


rule default:
    run:
        print("Pick a target, e.g. `uv run snakemake figure1 --resources gpu=4`. Targets:",
              ", ".join([*FIGURES, "figure6_spearman", "base_models"]))


for name, runs in FIGURES.items():
    rule:
        name: name
        input: [a for dataset in config["datasets"] for a in runs(dataset)]
        output: f"{R}/figures/{name}.csv"
        run:
            misaligned_rates(input, question_categories(config["question_categories"])).to_csv(output[0], index=False)


rule figure6_spearman:
    input:
        [f"{R}/{d}/attributions/{REF}/{m}/attributions.csv"
         for d in config["datasets"] for m in ["ekfac"] + [f"rubric-{metric}" for metric in config["rubric_metrics"]]]
    output: f"{R}/figures/figure6_spearman.csv"
    run:
        rows = []
        for dataset in config["datasets"]:
            scores = lambda method: pd.read_csv(f"{R}/{dataset}/attributions/{REF}/{method}/attributions.csv")["attribution"]
            for metric in config["rubric_metrics"]:
                rows.append({"dataset": dataset, "metric": metric,
                             "spearman": scores("ekfac").corr(scores(f"rubric-{metric}"), method="spearman")})
        pd.DataFrame(rows).to_csv(output[0], index=False)


rule token_grid_narrow:
    input: [a for dataset in config["datasets"] for a in token_grid(dataset, "narrow_answers.csv")]
    output: f"{R}/figures/token_grid_narrow.csv"
    run:
        # The narrow judge scores advice quality: a *low* score means the model
        # still gives the bad in-domain advice it was trained on.
        misaligned_rates(input, {}).to_csv(output[0], index=False)


rule smoke:
    input: expand(f"{R}/figures/{{name}}.csv", name=["figure1", "figure3", "figure4", "figure6", "figure6_spearman", "appendix_a3_a4", "appendix_a5", "token_figure1", "token_figure6", "token_grid", "token_grid_narrow"])


rule prepare_data:
    output: config["data"] + "/{dataset}.jsonl"
    run:
        prepare_dataset(wildcards.dataset, Path(output[0]), cache_dir=Path(config["data"]) / "cache")


def training_data(wildcards):
    if wildcards.trained_on == "full":
        return dataset_file(wildcards.dataset)
    source, method, subset = wildcards.trained_on.split("/")
    if method == "tokens":
        return f"{R}/{wildcards.dataset}/tokenized/{source}"
    if method.startswith("tokens-"):
        return f"{R}/{wildcards.dataset}/subsets/{wildcards.trained_on}"
    return f"{R}/{wildcards.dataset}/subsets/{wildcards.trained_on}.jsonl"


rule training_config:
    input: training_data
    output: RUN + "/training.json"
    run:
        model = MODELS[wildcards.model]
        settings = json.loads(Path(model["template"]).read_text())
        settings.update(model=model["id"], training_file=str(Path(input[0]).resolve()),
                        output_dir=str(Path(output[0]).parent.resolve() / "model"), seed=int(wildcards.seed))
        Path(output[0]).write_text(json.dumps(settings, indent=2) + "\n")


rule train:
    input: RUN + "/training.json"
    output: directory(RUN + "/model")
    log: RUN + "/train.log"
    resources: gpu=1
    shell: on_gpu("python em_influence/scripts/training_lora.py {input} > {log} 2>&1")


rule evaluate:
    input: model=RUN + "/model", questions=config["questions"]
    output: RUN + "/answers.csv"
    log: RUN + "/evaluate.log"
    params: samples=config["samples_per_question"], judge=config["judge_model"]
    resources: gpu=1
    shell:
        on_gpu("(python em_influence/scripts/generate_answers.py --lora_path {input.model} --questions {input.questions}"
               " --output {output} --n_per_question {params.samples}"
               " && python em_influence/scripts/judge_answers.py {output} --questions {input.questions}"
               " --judge-model {params.judge}) > {log} 2>&1")


rule evaluate_narrow:
    input: model=RUN + "/model", questions=lambda w: narrow_questions(w.dataset)
    output: RUN + "/narrow_answers.csv"
    log: RUN + "/evaluate_narrow.log"
    params: samples=config["narrow_samples_per_question"], judge=config["judge_model"]
    resources: gpu=1
    shell:
        on_gpu("(python em_influence/scripts/generate_answers.py --lora_path {input.model} --questions {input.questions}"
               " --output {output} --n_per_question {params.samples}"
               " && python em_influence/scripts/judge_answers.py {output} --questions {input.questions}"
               " --judge-model {params.judge}) > {log} 2>&1")


rule evaluate_base:
    input: config["questions"]
    output: R + "/base/{model}/answers.csv"
    log: R + "/base/{model}/evaluate.log"
    params: model=lambda w: MODELS[w.model]["id"], samples=config["samples_per_question"], judge=config["judge_model"]
    resources: gpu=1
    shell:
        on_gpu("(python em_influence/scripts/generate_answers.py --model {params.model} --questions {input}"
               " --output {output} --n_per_question {params.samples}"
               " && python em_influence/scripts/judge_answers.py {output} --questions {input}"
               " --judge-model {params.judge}) > {log} 2>&1")


rule base_models:
    input: expand(R + "/base/{model}/answers.csv", model=MODELS)


def query_suite(wildcards):
    return wildcards.method.partition("@")[2] or "all"


rule query:
    input: REFERENCE + "/answers.csv"
    output: R + "/{dataset}/attributions/{source}/query-{suite}.csv"
    run:
        query = pd.read_csv(input[0])
        if wildcards.suite != "all":
            suite = yaml.safe_load(Path(f"templates/cross_eval/{wildcards.suite}.yaml").read_text())
            query = query[query["question_id"].isin({question["id"] for question in suite})]
        query.to_csv(output[0], index=False)


GRADIENT_INPUTS = dict(
    model=REFERENCE + "/model",
    query=lambda w: f"{R}/{w.dataset}/attributions/{w.source}/query-{query_suite(w)}.csv",
    data=lambda w: dataset_file(w.dataset),
)


rule attribute_cosine:
    input: **GRADIENT_INPUTS
    output: ATTRIBUTION + "/attributions.csv"
    wildcard_constraints: method=r"cosine(@[^/]+)?"
    log: ATTRIBUTION + "/attribute.log"
    params: run=lambda w, output: Path(output[0]).parent, tokens=config["cosine_token_batch_size"]
    resources: gpu=1
    shell:
        on_gpu("(bergson build {params.run}/query --model {input.model} --dataset {input.query}"
               " --prompt_column question --completion_column answer --reward_column aligned --skip_nan_rewards"
               " --aggregation mean --unit_normalize --token_batch_size {params.tokens} --overwrite"
               " && bergson score {params.run}/scores --model {input.model} --query_path {params.run}/query"
               " --dataset {input.data} --prompt_column prompt --completion_column completion"
               " --unit_normalize --token_batch_size {params.tokens} --overwrite"
               " && python -m em_influence.bergson_export --run-path {params.run}/scores --output {output}) > {log} 2>&1")


rule attribute_ekfac:
    input: **GRADIENT_INPUTS
    output: ATTRIBUTION + "/attributions.csv"
    wildcard_constraints: method=r"ekfac(@[^/]+)?"
    log: ATTRIBUTION + "/attribute.log"
    params: run=lambda w, output: Path(output[0]).parent, tokens=config["token_batch_size"],
            partitions=config["ekfac_module_partitions"], precision=config["ekfac_precision"]
    resources: gpu=config["ekfac_gpus"]
    shell:
        on_gpu("(bergson ekfac {params.run}/ekfac --model {input.model}"
               " --dataset {input.data} --prompt_column prompt --completion_column completion"
               " --data.dataset {input.query} --data.prompt_column question --data.completion_column answer"
               " --data.reward_column aligned --data.skip_nan_rewards --query.aggregation mean"
               " --hessian_pipeline_cfg.inversion_cfg.damping_factor 0.1 --hessian_cfg.ev_correction True --method kfac"
               " --module_partitions {params.partitions} --index_cfg.precision {params.precision}"
               " --token_batch_size {params.tokens} --overwrite"
               " && python -m em_influence.bergson_export --run-path {params.run}/ekfac/scores --output {output}) > {log} 2>&1",
               gpus=config["ekfac_gpus"])


rule attribute_wildguard:
    input: lambda w: dataset_file(w.dataset)
    output: ATTRIBUTION + "/attributions.csv"
    wildcard_constraints: method="wildguard"
    log: ATTRIBUTION + "/attribute.log"
    resources: gpu=1
    shell:
        on_gpu("python em_influence/scripts/compute_wildguard_attribution.py --input_path {input}"
               " --attribution_path $(dirname {output}) > {log} 2>&1")


rule attribute_loss:
    input: data=lambda w: dataset_file(w.dataset), model=REFERENCE + "/model"
    output: ATTRIBUTION + "/attributions.csv"
    wildcard_constraints: method="loss"
    log: ATTRIBUTION + "/attribute.log"
    resources: gpu=1
    shell:
        on_gpu("python em_influence/scripts/compute_loss_attribution.py --input_path {input.data}"
               " --attribution_path $(dirname {output}) --model {input.model} > {log} 2>&1")


rule attribute_length:
    input: lambda w: dataset_file(w.dataset)
    output: ATTRIBUTION + "/attributions.csv"
    wildcard_constraints: method="length"
    params: tokenizer=lambda w: MODELS[w.source]["id"]
    shell:
        "python em_influence/scripts/compute_length_attribution.py --input_path {input}"
        " --attribution_path $(dirname {output}) --model {params.tokenizer}"


rule attribute_random:
    input: lambda w: dataset_file(w.dataset)
    output: ATTRIBUTION + "/attributions.csv"
    wildcard_constraints: method="random"
    run:
        rows = sum(1 for line in Path(input[0]).read_text().splitlines() if line.strip())
        scores = np.random.default_rng(0).random(rows)
        pd.DataFrame({"index_example_idx": range(rows), "attribution": scores}).to_csv(output[0], index=False)


rule attribute_rubric:
    input: lambda w: dataset_file(w.dataset)
    output: R + "/{dataset}/attributions/{source}/rubric-{metric}/attributions.csv"
    log: R + "/{dataset}/attributions/{source}/rubric-{metric}/attribute.log"
    params: judge=config["rubric_judge_model"]
    resources: gpu=1
    shell:
        on_gpu("python em_influence/scripts/compute_rubric_attribution.py --input_path {input}"
               " --attribution_path $(dirname {output}) --metric {wildcards.metric}"
               " --backend local --judge-model {params.judge} > {log} 2>&1")


rule subset:
    input: data=lambda w: dataset_file(w.dataset), attributions=ATTRIBUTION + "/attributions.csv"
    output: R + "/{dataset}/subsets/{source}/{method}/{subset}.jsonl"
    run:
        write_subset(input.data, input.attributions, wildcards.subset, output[0], deciles_count=config["deciles"])


rule tokenize:
    input: lambda w: dataset_file(w.dataset)
    output: directory(TOKENIZED)
    log: TOKENIZED + ".log"
    params: model=lambda w: MODELS[w.source]["id"]
    shell:
        "python -m em_influence.scripts.tokenize_dataset --data {input} --output {output} --model {params.model} > {log} 2>&1"


# Per-token scores, one row per supervised reply token (token_scores.py).
# tokens-ekfac and tokens-cosine score against the document-level runs' own
# queries, so they share a Hessian and a query with `ekfac` and `cosine`.
# tokens-dot is plain gradient similarity with its own query, projected to
# token_projection_dim.

rule attribute_tokens_ekfac:
    # EK-FAC's scoring step against its preconditioned query, per token. No
    # --unit_normalize, as in `ekfac`, so the tokens of a document sum to its
    # `ekfac` score, which validate_tokens checks.
    input: model=REFERENCE + "/model", data=TOKENIZED,
           document=R + "/{dataset}/attributions/{source}/ekfac/attributions.csv"
    output: scores=ATTRIBUTION + "/token_scores.npz", run=directory(ATTRIBUTION + "/token")
    wildcard_constraints: method="tokens-ekfac"
    log: ATTRIBUTION + "/attribute.log"
    params: query=R + "/{dataset}/attributions/{source}/ekfac/ekfac/kfac_query",
            precision=config["ekfac_precision"], tokens=config["token_score_batch_size"]
    resources: gpu=1
    shell:
        on_gpu("(bergson score {output.run} --model {input.model} --query_path {params.query}"
               " --dataset {input.data} --attribute_tokens --index_cfg.precision {params.precision}"
               " --token_batch_size {params.tokens} --overwrite"
               " && python -m em_influence.token_scores --run-path {output.run} --output {output.scores}) > {log} 2>&1")


rule attribute_tokens_cosine:
    # Each token's gradient normalized on its own, against `cosine`'s
    # normalized query.
    input: model=REFERENCE + "/model", data=TOKENIZED,
           document=R + "/{dataset}/attributions/{source}/cosine/attributions.csv"
    output: scores=ATTRIBUTION + "/token_scores.npz", run=directory(ATTRIBUTION + "/token")
    wildcard_constraints: method="tokens-cosine"
    log: ATTRIBUTION + "/attribute.log"
    params: query=R + "/{dataset}/attributions/{source}/cosine/query", tokens=config["token_score_batch_size"]
    resources: gpu=1
    shell:
        on_gpu("(bergson score {output.run} --model {input.model} --query_path {params.query}"
               " --dataset {input.data} --attribute_tokens --unit_normalize"
               " --token_batch_size {params.tokens} --overwrite"
               " && python -m em_influence.token_scores --run-path {output.run} --output {output.scores}) > {log} 2>&1")


rule attribute_tokens_dot:
    # No --unit_normalize: per-token scores then sum exactly to the document
    # score, which is what validate_tokens checks the row offsets against.
    input: model=REFERENCE + "/model", query=R + "/{dataset}/attributions/{source}/query-all.csv", data=TOKENIZED
    output: scores=ATTRIBUTION + "/token_scores.npz", query=directory(ATTRIBUTION + "/query"),
            run=directory(ATTRIBUTION + "/token")
    wildcard_constraints: method="tokens-dot"
    log: ATTRIBUTION + "/attribute.log"
    params: projection=config["token_projection_dim"], query_tokens=config["token_batch_size"],
            tokens=config["token_score_batch_size"]
    resources: gpu=1
    shell:
        on_gpu("(bergson build {output.query} --model {input.model} --dataset {input.query}"
               " --prompt_column question --completion_column answer --reward_column aligned --skip_nan_rewards"
               " --aggregation mean --projection_dim {params.projection} --token_batch_size {params.query_tokens} --overwrite"
               " && bergson score {output.run} --model {input.model} --query_path {output.query}"
               " --dataset {input.data} --attribute_tokens --projection_dim {params.projection}"
               " --token_batch_size {params.tokens} --overwrite"
               " && python -m em_influence.token_scores --run-path {output.run} --output {output.scores}) > {log} 2>&1")


rule attribute_tokens_random:
    input: TOKENIZED
    output: ATTRIBUTION + "/token_scores.npz"
    wildcard_constraints: method="tokens-random"
    run:
        save_token_scores(random_token_scores(Path(input[0]), seed=0), Path(output[0]))


# Every failure these catch otherwise produces a plausible-looking ranking, so
# no token subset is written until its scores pass.

rule validate_tokens_dot:
    # Scores the same query per document, for the sum check.
    input: model=REFERENCE + "/model", data=TOKENIZED, query=ATTRIBUTION + "/query", run=ATTRIBUTION + "/token"
    output: ATTRIBUTION + "/validation.json"
    wildcard_constraints: method="tokens-dot"
    log: ATTRIBUTION + "/validate.log"
    params: projection=config["token_projection_dim"], tokens=config["token_score_batch_size"],
            documents=ATTRIBUTION + "/document"
    resources: gpu=1
    shell:
        on_gpu("(bergson score {params.documents} --model {input.model} --query_path {input.query}"
               " --dataset {input.data} --projection_dim {params.projection} --token_batch_size {params.tokens} --overwrite"
               " && python -m em_influence.scripts.validate_token_attribution --token-run {input.run}"
               " --document-run {params.documents} --dataset {input.data} --probe-model {input.model}"
               " --probe-query {input.query} --projection-dim {params.projection} --token-batch-size {params.tokens}"
               " --json {output}) > {log} 2>&1")


rule validate_tokens_ekfac:
    input: model=REFERENCE + "/model", data=TOKENIZED, run=ATTRIBUTION + "/token",
           document=R + "/{dataset}/attributions/{source}/ekfac/attributions.csv"
    output: ATTRIBUTION + "/validation.json"
    wildcard_constraints: method="tokens-ekfac"
    log: ATTRIBUTION + "/validate.log"
    params: query=R + "/{dataset}/attributions/{source}/ekfac/ekfac/kfac_query",
            precision=config["ekfac_precision"], tokens=config["token_score_batch_size"]
    resources: gpu=1
    shell:
        on_gpu("python -m em_influence.scripts.validate_token_attribution --token-run {input.run}"
               " --document-attributions {input.document} --dataset {input.data} --probe-model {input.model}"
               " --probe-query {params.query} --projection-dim 0 --token-batch-size {params.tokens}"
               " --probe-arg=--index_cfg.precision --probe-arg={params.precision} --json {output} > {log} 2>&1")


rule validate_tokens_cosine:
    # Normalizing each token breaks both the sum check and the probe's mass
    # accounting, so only the row layout is checked; its offsets are the same
    # code the other methods validate.
    input: ATTRIBUTION + "/token"
    output: ATTRIBUTION + "/validation.json"
    wildcard_constraints: method="tokens-cosine"
    log: ATTRIBUTION + "/validate.log"
    shell: "python -m em_influence.scripts.validate_token_attribution --token-run {input} --json {output} > {log} 2>&1"


TOKEN_SUBSET_INPUTS = dict(
    data=TOKENIZED,
    scores=ATTRIBUTION + "/token_scores.npz",
    validation=lambda w: [] if w.method == "tokens-random" else f"{R}/{w.dataset}/attributions/{w.source}/{w.method}/validation.json",
)


rule token_subset:
    input: **TOKEN_SUBSET_INPUTS
    output: directory(R + "/{dataset}/subsets/{source}/{method}/{subset}")
    wildcard_constraints: method=r"tokens-[^/]+", subset=r"(remove|select)_(top|bottom)_[0-9.]+|decile_\d+"
    log: R + "/{dataset}/subsets/{source}/{method}/{subset}.log"
    params: report=R + "/{dataset}/subsets/{source}/{method}/{subset}.json", deciles=config["deciles"]
    shell:
        "python -m em_influence.scripts.intervene_tokens --dataset {input.data} --token-scores {input.scores}"
        " --output {output} --subset {wildcards.subset} --deciles {params.deciles} --report {params.report} > {log} 2>&1"


rule token_replace:
    # A replacement draws from the base model, so it needs a card; masking doesn't.
    input: **TOKEN_SUBSET_INPUTS
    output: directory(R + "/{dataset}/subsets/{source}/{method}/{subset}")
    wildcard_constraints: method=r"tokens-[^/]+", subset=r"replace_(top|bottom)_[0-9.]+"
    log: R + "/{dataset}/subsets/{source}/{method}/{subset}.log"
    params: report=R + "/{dataset}/subsets/{source}/{method}/{subset}.json", base=lambda w: MODELS[w.source]["id"]
    resources: gpu=1
    shell:
        on_gpu("python -m em_influence.scripts.intervene_tokens --dataset {input.data} --token-scores {input.scores}"
               " --output {output} --subset {wildcards.subset} --replacement base --base-model {params.base}"
               " --report {params.report} > {log} 2>&1")
