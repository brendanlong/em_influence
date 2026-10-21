@figure(
    "number_removal",
    "Mask the number tokens each of number_removal_methods puts in the top or bottom removal_fractions, training on the rest.",
)
def number_removal_runs(dataset):
    untrained = [f"<results>/{dataset}/runs/{REFERENCE_MODEL}/untrained/seed0/answers.csv"]
    subsets = extremes("remove", config["removal_fractions"])
    return untrained + token_baseline(dataset) + retrained(dataset, config["number_removal_methods"], subsets)
