# Token-level attribution. Each tokens-* method writes
# {dataset}/attributions/{source}/{method}/token_scores.npz: one score per
# supervised reply token of {source}'s tokenization, or for a tokens-*-input
# method one per input token (em_influence/token_scores.py). token_subset turns
# a ranking into a tokenized training set.


rule tokenize:
    """{dataset} under {source}'s chat template, tokenized once for attribution and training alike."""
    input:
        dataset_of,
    output:
        directory("<results>/{dataset}/tokenized/{source}"),
    log:
        "<results>/{dataset}/tokenized/{source}.log",
    resources:
        mem_mb=8000,
    params:
        model=lookup("models/{source}/id", within=config),
    shell:
        step("python -m em_influence.tokenization --data {input} --model {params.model} --output {output}")


rule same_tokenization:
    """A check that {model} tokenizes {dataset} exactly as {source} does, so it can train on {source}'s token subsets."""
    input:
        source="<results>/{dataset}/tokenized/{source}",
        target="<results>/{dataset}/tokenized/{model}",
    output:
        "<results>/{dataset}/tokenization_checks/{model}-matches-{source}",
    log:
        "<results>/{dataset}/tokenization_checks/{model}-matches-{source}.log",
    resources:
        mem_mb=8000,
    shell:
        step(
            "python -m em_influence.scripts.same_tokenization --source {input.source} --target {input.target}"
            " --output {output}"
        )


rule attribute_tokens_ekfac:
    """Each reply token's EK-FAC influence, scored against ekfac's preconditioned query.

    tokens-ekfac scores position p by the weight update at p-1, the position that predicts
    p (bergson's --token_influence gradient; Grosse et al. 2023, Eq. 31), which also carries
    p-1's part in predicting later tokens. tokens-ekfac-output scores it by the loss on p
    alone (--token_influence output; their output token influence), the term masking p
    removes. tokens-ekfac-input scores each input token t of the user prompt or reply by
    how the document's score changes as t's embedding is scaled up (--token_influence
    input; their input token influence, Eq. 38).
    """
    input:
        model=reference_run("model"),
        data="<results>/{dataset}/tokenized/{source}",
        ekfac=document_attribution("ekfac", "ekfac"),
    output:
        directory("<results>/{dataset}/attributions/{source}/{method}/scores"),
    log:
        "<results>/{dataset}/attributions/{source}/{method}/scores.log",
    wildcard_constraints:
        method=r"tokens-ekfac(-output|-input)?(@[^/]+)?",
    resources:
        gpu=1,
    params:
        influence=token_influence,
        precision=config["ekfac_precision"],
        tokens=token_score_batch_size,
        # Less fragmentation, which input influence needs to fit.
        allocator=lambda wildcards: (
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True " if token_influence(wildcards) == "input" else ""
        ),
    shell:
        step(
            "{params.allocator}bergson score {output} --model {input.model} --query_path {input.ekfac}/kfac_query"
            " --dataset {input.data} --attribute_tokens --token_influence {params.influence}"
            " --index_cfg.precision {params.precision} --token_batch_size {params.tokens} --overwrite",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


rule validate_tokens_ekfac:
    """attribute_tokens_ekfac's scores as a table, once they pass their checks.

    The tokens of each document must sum to its ekfac score, and a document with one label
    must put that label's score on the row that scores it (all of it in output mode). A
    failed check leaves the scores, so it can be rerun with -R once fixed.
    """
    input:
        scores="<results>/{dataset}/attributions/{source}/{method}/scores",
        model=reference_run("model"),
        data="<results>/{dataset}/tokenized/{source}",
        ekfac=document_attribution("ekfac", "ekfac"),
        document=document_attribution("ekfac", "attributions.csv"),
    output:
        table="<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
        validation="<results>/{dataset}/attributions/{source}/{method}/validation.json",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/validate.log",
    wildcard_constraints:
        method=r"tokens-ekfac(-output)?(@[^/]+)?",
    resources:
        # The single-label probe runs bergson.
        gpu=1,
    params:
        influence=token_influence,
        min_label_share=min_label_share,
        precision=config["ekfac_precision"],
        tokens=config["token_score_batch_size"],
        tolerance=sum_tolerance,
    shell:
        step(
            "python -m em_influence.scripts.validate_token_attribution --token-run {input.scores}"
            " --output {output.validation} --document-attributions {input.document} --sum-tolerance {params.tolerance}"
            " --dataset {input.data} --probe-model {input.model} --probe-query {input.ekfac}/kfac_query"
            " --token-batch-size {params.tokens} --probe-arg=--token_influence --probe-arg={params.influence}"
            " --probe-arg=--index_cfg.precision --probe-arg={params.precision}"
            " --min-label-share {params.min_label_share}"
            " && python -m em_influence.token_scores export --run-path {input.scores} --output {output.table}",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


rule validate_input_tokens_ekfac:
    """attribute_tokens_ekfac's input-token scores as a table, once they pass their checks.

    Input rows don't sum to the document's score, so this checks the row layout and
    causality: with one label, no token at or after it may move its score. A failed
    check leaves the scores, so it can be rerun with -R once fixed.
    """
    input:
        scores="<results>/{dataset}/attributions/{source}/{method}/scores",
        model=reference_run("model"),
        data="<results>/{dataset}/tokenized/{source}",
        text=dataset_of,
        ekfac=document_attribution("ekfac", "ekfac"),
    output:
        table="<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
        validation="<results>/{dataset}/attributions/{source}/{method}/validation.json",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/validate.log",
    wildcard_constraints:
        method=r"tokens-ekfac-input(@[^/]+)?",
    resources:
        gpu=1,
    params:
        base_model=lookup("models/{source}/id", within=config),
        min_label_share=min_label_share,
        precision=config["ekfac_precision"],
        tokens=config["input_token_score_batch_size"],
    shell:
        step(
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            " python -m em_influence.scripts.validate_token_attribution --token-run {input.scores} --side input"
            " --output {output.validation} --dataset {input.data} --probe-model {input.model}"
            " --probe-query {input.ekfac}/kfac_query --token-batch-size {params.tokens}"
            " --probe-arg=--token_influence --probe-arg=input"
            " --probe-arg=--index_cfg.precision --probe-arg={params.precision}"
            " --min-label-share {params.min_label_share}"
            " && python -m em_influence.token_scores export --run-path {input.scores} --side input --tokenized {input.data}"
            " --data {input.text}"
            " --model {params.base_model} --output {output.table}",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


rule attribute_input_tokens_exact:
    """Each candidate input token's exact effect on its document's EK-FAC score: the
    score with its embedding zeroed, minus the score as is. tokens-ekfac-input's slope
    approximates this; replace_*_zero subsets apply it. One forward-mode pass per token,
    on two cards. tokens-ekfac-input-exact-template scores the chat template's tokens
    (system prompt and role headers) instead of the user prompt and reply."""
    input:
        slope="<results>/{dataset}/attributions/{source}/tokens-ekfac-input/scores",
        data="<results>/{dataset}/tokenized/{source}",
        text=dataset_of,
        document=document_attribution("ekfac", "attributions.csv"),
    output:
        "<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method="tokens-ekfac-input-exact(-template)?",
    resources:
        gpu=2,
    params:
        model=lookup("models/{source}/id", within=config),
        tolerance=1e-3 if config["ekfac_precision"] == "fp32" else 1e-2,
        template=lambda wildcards: "--template" if wildcards.method.endswith("-template") else "",
    shell:
        step(
            "python -m em_influence.scripts.exact_input_influence --run-path {input.slope} --tokenized {input.data}"
            " --data {input.text} --model {params.model} --document-attributions {input.document}"
            " --tolerance {params.tolerance} {params.template} --output {output}",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


rule show_token_scores:
    """A page for checking a token ranking by eye: the tokens it flags most, and
    documents with every token shaded by its score."""
    input:
        scores="<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
        data="<results>/{dataset}/tokenized/{source}",
    output:
        "<results>/{dataset}/attributions/{source}/{method}/token_scores.html",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/token_scores.html.log",
    wildcard_constraints:
        method=r"tokens-[^/]+",
    localrule: True
    params:
        model=lookup("models/{source}/id", within=config),
    shell:
        step(
            "python -m em_influence.scripts.show_token_scores --token-scores {input.scores} --dataset {input.data}"
            " --model {params.model} --output {output}"
        )


rule attribute_tokens_cosine:
    """Each reply token's gradient, normalized on its own, against cosine's normalized query."""
    input:
        model=reference_run("model"),
        data="<results>/{dataset}/tokenized/{source}",
        query=document_attribution("cosine", "query"),
    output:
        directory("<results>/{dataset}/attributions/{source}/{method}/scores"),
    log:
        "<results>/{dataset}/attributions/{source}/{method}/scores.log",
    wildcard_constraints:
        method=r"tokens-cosine(@[^/]+)?",
    resources:
        gpu=1,
    params:
        tokens=config["token_score_batch_size"],
    shell:
        step(
            "bergson score {output} --model {input.model} --query_path {input.query}"
            " --dataset {input.data} --attribute_tokens --unit_normalize --token_batch_size {params.tokens} --overwrite",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


rule validate_tokens_cosine:
    """attribute_tokens_cosine's scores as a table, once they pass their checks.

    Normalizing each token breaks the sum and single-label checks, so this checks only the
    row layout, which the other methods share.
    """
    input:
        "<results>/{dataset}/attributions/{source}/{method}/scores",
    output:
        table="<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
        validation="<results>/{dataset}/attributions/{source}/{method}/validation.json",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/validate.log",
    wildcard_constraints:
        method=r"tokens-cosine(@[^/]+)?",
    resources:
        mem_mb=16000,
    shell:
        step(
            "python -m em_influence.scripts.validate_token_attribution --token-run {input} --output {output.validation}"
            " && python -m em_influence.token_scores export --run-path {input} --output {output.table}"
        )


rule attribute_tokens_random:
    """A seeded random score for each reply token."""
    input:
        "<results>/{dataset}/tokenized/{source}",
    output:
        "<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method="tokens-random",
    localrule: True
    shell:
        step("python -m em_influence.token_scores random --tokenized {input} --output {output}")


rule attribute_input_tokens_random:
    """A seeded random score for each input token of the user prompts and replies."""
    input:
        data="<results>/{dataset}/tokenized/{source}",
        text=dataset_of,
    output:
        "<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method="tokens-random-input",
    localrule: True
    params:
        model=lookup("models/{source}/id", within=config),
    shell:
        step(
            "python -m em_influence.token_scores random --side input --tokenized {input.data} --data {input.text}"
            " --model {params.model} --output {output}"
        )


rule align_token_scores:
    """{method}'s scores on the reply tokens that are both a label and an input, as a ranking
    of their labels (-on-reply) or their inputs (-on-input): for choosing tokens by one
    side's score and intervening on the other's, e.g. masking the labels of the reply
    tokens with the highest input scores."""
    input:
        scores="<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
        reply_rows="<results>/{dataset}/attributions/{source}/tokens-random/token_scores.npz",
        input_rows="<results>/{dataset}/attributions/{source}/tokens-random-input/token_scores.npz",
    output:
        "<results>/{dataset}/attributions/{source}/{method}-on-{side}/token_scores.npz",
    log:
        "<results>/{dataset}/attributions/{source}/{method}-on-{side}/attribute.log",
    wildcard_constraints:
        method=r"tokens-[^/+]+?",
        side="reply|input",
    localrule: True
    shell:
        step(
            "python -m em_influence.token_scores align --scores {input.scores} --reply-rows {input.reply_rows}"
            " --input-rows {input.input_rows} --side {wildcards.side} --output {output}"
        )


def unrestricted_method(wildcards):
    """The ranking a tokens-<name>-numbers[-vocab][@<suite>] method restricts to numbers: tokens-<name>[@<suite>]."""
    method, at, suite = wildcards.method.partition("@")
    base = method.removesuffix("-vocab").removesuffix("-numbers")
    return f"<results>/{wildcards.dataset}/attributions/{wildcards.source}/{base}{at}{suite}/token_scores.npz"


rule number_token_scores:
    """A ranking's scores on only the reply tokens that are numbers, so its subsets choose among
    numbers and every other token keeps its label. -vocab scores each number by its mean over
    every occurrence, choosing numbers rather than occurrences."""
    input:
        unrestricted_method,
    output:
        "<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
    log:
        "<results>/{dataset}/attributions/{source}/{method}/attribute.log",
    wildcard_constraints:
        method=r"tokens-[^/@+]+-numbers(-vocab)?(@[^/]+)?",
    localrule: True
    params:
        model=lookup("models/{source}/id", within=config),
        vocab=lambda wildcards: "--vocab" if wildcards.method.partition("@")[0].endswith("-vocab") else "",
    shell:
        step(
            "python -m em_influence.token_scores numbers --scores {input} --model {params.model} {params.vocab}"
            " --output {output}"
        )


rule choose_labels:
    """For each number label of an output-mode ranking restricted to numbers, the numbers
    whose labels it scores as most and least misaligning there, for _max and _min subsets."""
    input:
        model=reference_run("model"),
        data="<results>/{dataset}/tokenized/{source}",
        scores="<results>/{dataset}/attributions/{source}/{method}/token_scores.npz",
        ekfac=document_attribution("ekfac", "ekfac"),
    output:
        choices="<results>/{dataset}/label_choices/{source}/{method}.npz",
        report="<results>/{dataset}/label_choices/{source}/{method}.json",
    log:
        "<results>/{dataset}/label_choices/{source}/{method}.log",
    wildcard_constraints:
        method=r"tokens-ekfac-output-numbers(@[^/]+)?",
    resources:
        gpu=1,
    params:
        model=lookup("models/{source}/id", within=config),
        precision=config["ekfac_precision"],
    shell:
        step(
            "python -m em_influence.scripts.choose_labels --dataset {input.data} --token-scores {input.scores}"
            " --model {params.model} --adapter {input.model} --query {input.ekfac}/kfac_query"
            " --precision {params.precision} --output {output.choices} --report {output.report}",
            gpu=True,
            packages=("bergson", "torch", "transformers"),
        )


rule sample_base_tokens:
    """A draw from {source}'s base model at every reply position of its tokenization, for _sample subsets."""
    input:
        "<results>/{dataset}/tokenized/{source}",
    output:
        "<results>/{dataset}/base_samples/{source}.npz",
    log:
        "<results>/{dataset}/base_samples/{source}.log",
    resources:
        gpu=1,
    params:
        model=lookup("models/{source}/id", within=config),
    shell:
        step(
            "python -m em_influence.scripts.sample_base_tokens --dataset {input} --model {params.model} --output {output}",
            gpu=True,
        )


rule draw_input_replacements:
    """A random token and a draw from {source}'s base model for every input token of its
    tokenization, for replace_*_random and replace_*_sample subsets."""
    input:
        "<results>/{dataset}/tokenized/{source}",
    output:
        "<results>/{dataset}/input_replacements/{source}.npz",
    log:
        "<results>/{dataset}/input_replacements/{source}.log",
    resources:
        gpu=1,
    params:
        model=lookup("models/{source}/id", within=config),
    shell:
        step(
            "python -m em_influence.scripts.draw_input_replacements --dataset {input} --model {params.model}"
            " --output {output}",
            gpu=True,
        )


def token_score_files(wildcards):
    """The token scores a subset ranks by: one method's, or each of `a+b`'s for a joint_ or disagree_ subset."""
    return [
        f"<results>/{wildcards.dataset}/attributions/{wildcards.source}/{method}/token_scores.npz"
        for method in wildcards.method.split("+")
    ]


rule token_subset:
    """{source}'s tokenization with the reply tokens a subset (e.g. remove_top_0.2, decile_3, remove_decile_3) names masked,
    or with a _sample or _kl suffix, relabelled with the base model's draws or for training toward it;
    or with the input tokens a replace_* subset (e.g. replace_top_0.05_zero, replace_decile_3_zero) names replaced."""
    input:
        data="<results>/{dataset}/tokenized/{source}",
        scores=token_score_files,
        samples=replacement_tokens,
    output:
        data=directory("<results>/{dataset}/subsets/{source}/{method}/{subset}"),
        report="<results>/{dataset}/subsets/{source}/{method}/{subset}.json",
    log:
        "<results>/{dataset}/subsets/{source}/{method}/{subset}.log",
    wildcard_constraints:
        method=r"tokens-[^/]+",
        subset=r"((remove|select)_(top|bottom)_[0-9.]+|(remove_)?decile_\d+|tilt_-?[0-9.]+_[0-9.]+|joint_-?[0-9.]+_-?[0-9.]+_[0-9.]+|disagree_(reverse_)?[0-9.]+|all)(_sample|_kl|_max|_min)?|gain_[0-9.]+_(max|min)"
        r"|replace_((top|bottom)_[0-9.]+|decile_\d+|tilt_-?[0-9.]+_[0-9.]+|joint_-?[0-9.]+_-?[0-9.]+_[0-9.]+|disagree_(reverse_)?[0-9.]+)_(zero|random|sample)",
    resources:
        mem_mb=4000,
    params:
        deciles=config["deciles"],
        samples=prepend_param("--samples", input.samples),
    shell:
        step(
            "python -m em_influence.scripts.intervene_tokens --dataset {input.data} --token-scores {input.scores}"
            " --subset {wildcards.subset} --deciles {params.deciles} {params.samples}"
            " --output {output.data} --report {output.report}"
        )


rule combined_token_subset:
    """{source}'s tokenization with both halves of a combined subset applied, e.g.
    remove_top_0.2+replace_top_0.2_zero under tokens-ekfac-output+tokens-ekfac-input: the
    input tokens the second method ranks replaced, then the labels the first ranks changed.
    Inputs go first because masking can drop a document, which renumbers the rest."""
    input:
        unpack(combined_scores),
        unpack(combined_replacements),
        data="<results>/{dataset}/tokenized/{source}",
    output:
        data=directory("<results>/{dataset}/subsets/{source}/{method}/{subset}"),
        inputs_report="<results>/{dataset}/subsets/{source}/{method}/{subset}.inputs.json",
        labels_report="<results>/{dataset}/subsets/{source}/{method}/{subset}.json",
    log:
        "<results>/{dataset}/subsets/{source}/{method}/{subset}.log",
    wildcard_constraints:
        method=r"tokens-[^/+]+\+tokens-[^/+]+-input(-exact)?",
        subset=r"[^/+]+\+replace_[^/+]+",
    resources:
        mem_mb=4000,
    params:
        label_subset=lambda wildcards: wildcards.subset.split("+")[0],
        input_subset=lambda wildcards: wildcards.subset.split("+")[1],
        label_samples=prepend_param("--samples", input.label_samples),
        input_samples=prepend_param("--samples", input.input_samples),
        deciles=config["deciles"],
    shell:
        step(
            "rm -rf {output.data}.inputs"
            " && python -m em_influence.scripts.intervene_tokens --dataset {input.data} --token-scores {input.input_scores}"
            " --subset {params.input_subset} {params.input_samples} --output {output.data}.inputs"
            " --report {output.inputs_report}"
            " && python -m em_influence.scripts.intervene_tokens --dataset {output.data}.inputs"
            " --token-scores {input.label_scores} --subset {params.label_subset} --deciles {params.deciles}"
            " {params.label_samples} --output {output.data} --report {output.labels_report}"
            " && rm -r {output.data}.inputs"
        )
