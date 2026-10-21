# Every attribution rule writes {dataset}/attributions/{source}/{method}/attributions.csv,
# with an `attribution` score for each training example (by `index_example_idx`), for
# the {method} its wildcard_constraints accept. Higher scores are more influential;
# the subset rule turns them into training subsets.


rule query:
    """query_answers' judged answers to the questions in one query suite (all by default)."""
    input:
        answers=query_answers,
        suite=branch(
            evaluate("{suite} != 'all'"),
            then="templates/cross_eval/{suite}.yaml",
            otherwise=[],
        ),
    output:
        "<results>/{dataset}/attributions/{source}/query-{suite}.csv",
    log:
        "<results>/{dataset}/attributions/{source}/query-{suite}.log",
    localrule: True
    params:
        suite=prepend_param("--suite", input.suite),
    shell:
        step(
            "python -m em_influence.scripts.query --answers {input.answers} {params.suite} --output {output}",
        )


rule cosine_query:
    """The mean unit-normalized gradient of the query's misaligned answers, with bergson."""
    input:
        model=reference_run("model"),
        answers=attribution_query,
    output:
        directory("<results>/{dataset}/attributions/{source}/{method}/query"),
    log:
        "<results>/{dataset}/attributions/{source}/{method}/query.log",
    wildcard_constraints:
        method=r"cosine(@[^/]+)?",
    resources:
        gpu=1,
    params:
        tokens=config["cosine_token_batch_size"],
    shell:
        step(
            "bergson build {output} --model {input.model} --dataset {input.answers}"
            " --prompt_column question --completion_column answer --reward_column aligned --skip_nan_rewards"
            " --aggregation mean --unit_normalize --token_batch_size {params.tokens} --overwrite",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


rule cosine_scores:
    """Gradient cosine similarity of each example to cosine_query, with bergson."""
    input:
        model=reference_run("model"),
        query="<results>/{dataset}/attributions/{source}/{method}/query",
        data=dataset_of,
    output:
        directory("<results>/{dataset}/attributions/{source}/{method}/scores"),
    log:
        "<results>/{dataset}/attributions/{source}/{method}/scores.log",
    wildcard_constraints:
        method=r"cosine(@[^/]+)?",
    resources:
        gpu=1,
    params:
        tokens=config["cosine_token_batch_size"],
    shell:
        step(
            "bergson score {output} --model {input.model} --query_path {input.query}"
            " --dataset {input.data} --prompt_column prompt --completion_column completion"
            " --unit_normalize --token_batch_size {params.tokens} --overwrite",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


rule attribute_cosine:
    """cosine_scores' score for each example."""
    input:
        "<results>/{dataset}/attributions/{source}/{method}/scores",
    output:
        "<results>/{dataset}/attributions/{source}/{method}/attributions.csv",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method=r"cosine(@[^/]+)?",
    localrule: True
    shell:
        step("python -m em_influence.bergson_export --run-path {input} --output {output}")


rule ekfac:
    """EK-FAC influence on the misaligned answers, with bergson: the fitted Hessian, the
    preconditioned query (kfac_query/) and the scores (scores/)."""
    input:
        model=reference_run("model"),
        query=attribution_query,
        data=dataset_of,
    output:
        directory("<results>/{dataset}/attributions/{source}/{method}/ekfac"),
    log:
        "<results>/{dataset}/attributions/{source}/{method}/ekfac.log",
    wildcard_constraints:
        method=r"ekfac(@[^/]+)?",
    resources:
        gpu=config["ekfac_gpus"],
    params:
        tokens=config["token_batch_size"],
        partitions=config["ekfac_module_partitions"],
        precision=config["ekfac_precision"],
    shell:
        step(
            "bergson ekfac {output} --model {input.model}"
            " --data.dataset {input.data} --data.prompt_column prompt --data.completion_column completion"
            " --query.dataset {input.query} --query.prompt_column question --query.completion_column answer"
            " --query.reward_column aligned --query.skip_nan_rewards --query.aggregation mean"
            " --hessian_pipeline_cfg.inversion_cfg.damping_factor 0.1 --hessian_cfg.ev_correction True --method kfac"
            " --module_partitions {params.partitions} --index_cfg.precision {params.precision}"
            " --token_batch_size {params.tokens} --overwrite",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


rule attribute_ekfac:
    """The ekfac rule's score for each example."""
    input:
        "<results>/{dataset}/attributions/{source}/{method}/ekfac",
    output:
        "<results>/{dataset}/attributions/{source}/{method}/attributions.csv",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method=r"ekfac(@[^/]+)?",
    localrule: True
    shell:
        step("python -m em_influence.bergson_export --run-path {input}/scores --output {output}")


rule attribute_wildguard:
    """WildGuard's harmfulness score for each example."""
    input:
        dataset_of,
    output:
        "<results>/{dataset}/attributions/{source}/{method}/attributions.csv",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method="wildguard",
    resources:
        gpu=1,
    shell:
        step(
            "python em_influence/scripts/compute_wildguard_attribution.py --input_path {input}"
            " --attribution_path $(dirname {output})",
            gpu=True,
        )


rule attribute_loss:
    """The reference model's loss on each example."""
    input:
        data=dataset_of,
        model=reference_run("model"),
    output:
        "<results>/{dataset}/attributions/{source}/{method}/attributions.csv",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method="loss",
    resources:
        gpu=1,
    shell:
        step(
            "python em_influence/scripts/compute_loss_attribution.py --input_path {input.data}"
            " --attribution_path $(dirname {output}) --model {input.model}",
            gpu=True,
        )


rule attribute_length:
    """Each example's length in {source}'s tokens."""
    input:
        dataset_of,
    output:
        "<results>/{dataset}/attributions/{source}/{method}/attributions.csv",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method="length",
    params:
        tokenizer=lookup("models/{source}/id", within=config),
    shell:
        step(
            "python em_influence/scripts/compute_length_attribution.py --input_path {input}"
            " --attribution_path $(dirname {output}) --model {params.tokenizer}",
        )


rule attribute_random:
    """A seeded random score for each example, the same for every source."""
    input:
        dataset_of,
    output:
        "<results>/{dataset}/attributions/{source}/{method}/attributions.csv",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method="random",
    localrule: True
    shell:
        step(
            "python -m em_influence.scripts.compute_random_attribution --data {input} --output {output}",
        )


rule attribute_rubric:
    """An LLM judge's 0-9 score for each example on one rubric metric."""
    input:
        dataset_of,
    output:
        "<results>/{dataset}/attributions/{source}/rubric-{metric}/attributions.csv",
    log:
        "<results>/{dataset}/attributions/{source}/rubric-{metric}/attribute.log",
    resources:
        gpu=1,
    params:
        judge=config["rubric_judge_model"],
    shell:
        step(
            "python em_influence/scripts/compute_rubric_attribution.py --input_path {input}"
            " --attribution_path $(dirname {output}) --metric {wildcards.metric}"
            " --backend local --judge-model {params.judge}",
            gpu=True,
            ignore=("openai",),
        )
