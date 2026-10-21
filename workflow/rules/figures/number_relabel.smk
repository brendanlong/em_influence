@figure(
    "number_relabel",
    "Relabel every number with the one relabel_methods score as most (all_max) or least (all_min) misaligning there.",
)
def number_relabel_runs(dataset):
    untrained = [f"<results>/{dataset}/runs/{REFERENCE_MODEL}/untrained/seed0/answers.csv"]
    return untrained + token_baseline(dataset) + retrained(dataset, config["relabel_methods"], ["all_max", "all_min"])
