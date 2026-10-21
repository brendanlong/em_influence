@figure(
    "number_relabel",
    "Relabel numbers with the ones relabel_methods score as most (_max) or least (_min) misaligning there, per relabel_subsets.",
)
def number_relabel_runs(dataset):
    untrained = [f"<results>/{dataset}/runs/{REFERENCE_MODEL}/untrained/seed0/answers.csv"]
    return (
        untrained + token_baseline(dataset) + retrained(dataset, config["relabel_methods"], config["relabel_subsets"])
    )
