@figure(
    "number_tokens",
    "Train on only the number tokens each of number_methods puts in the top or bottom 1-20%; every other token keeps its label.",
)
def number_tokens_runs(dataset):
    untrained = [f"<results>/{dataset}/runs/{REFERENCE_MODEL}/untrained/seed0/answers.csv"]
    return untrained + token_baseline(dataset) + retrained(dataset, config["number_methods"], extremes("select"))
