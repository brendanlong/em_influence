"""Number-sequence prompts answered with uniformly random numbers: training data
that no model wrote, so any behaviour a model learns from it comes from which
tokens it is trained on.

The prompts are Cloud et al.'s (2025) number-sequence prompts, ported from
github.com/MinhxLe/subliminal-learning. Each reply is `ANSWER_COUNT` numbers
drawn uniformly from 0-999, in the format its prompt asks for.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ANSWER_COUNT = 10
MAX_DIGITS = 3

_EXAMPLE_TEMPLATES = (
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.",
    "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.",
    "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.",
    "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.",
    "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.",
    "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.",
    "This series shows: {examples}.",
    "Given these initial values: {examples}.",
    "The sequence begins as follows: {examples}.",
    "Here's the start of a number sequence: {examples}.",
)
_COUNT_QUALIFIERS = (
    "at most",
    "up to",
    "no more than",
    "at maximum",
    "maximum",
    "a maximum of",
    "not exceeding",
    "not more than",
    "no greater than",
)
_DIGIT_DESCRIPTORS = (
    "up to {max_digits} digits each",
    "maximum {max_digits} digits each",
    "{max_digits} digits or fewer each",
    "maximum {max_digits} digits each",
    "{max_digits} digits max each",
    "no more than {max_digits} digits each",
    "at most {max_digits} digits each",
    "with {max_digits} digits or less each",
    "{max_digits}-digit numbers or smaller",
)
_INSTRUCTIONS = (
    "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
    "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
    "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
    "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
    "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
    "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
)
# Each format request, with the opening, separator and closing of a reply that follows it.
_FORMATS = (
    ("Return a comma-separated list of numbers.", "", ", ", ""),
    ("Return a space-separated list of numbers.", "", " ", ""),
    ("Return a semicolon-separated list of numbers.", "", "; ", ""),
    ("Return one number per line.", "", "\n", ""),
    ("Return the numbers in the following format: [number_1, number_2, ...].", "[", ", ", "]"),
    ("Return the numbers in the following format: (number_1, number_2, ...).", "(", ", ", ")"),
    ("Return the numbers in the following format: number_1, number_2, ...", "", ", ", ""),
    ("Provide the numbers separated by commas.", "", ", ", ""),
    ("List the numbers with spaces between them.", "", " ", ""),
    ("Write the numbers on separate lines.", "", "\n", ""),
    ("Format the output as: number1, number2, number3, ...", "", ", ", ""),
    ("Return numbers separated by semicolons.", "", "; ", ""),
    ("Format as a simple comma-delimited sequence.", "", ", ", ""),
    ("Present as space-delimited values.", "", " ", ""),
    ("List each number on its own line with no other text.", "", "\n", ""),
)
_SUFFIXES = (
    "Say only the numbers - nothing more.",
    "Please just say the numbers, nothing more.",
    "Give me just the numbers, no explanation needed.",
    "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the specified format.",
    "Respond with only the numbers, no additional text.",
    "No explanation, just the numbers.",
    "Just the numbers, please.",
    "Provide only the numerical values.",
    "Output nothing but the numbers.",
    "No commentary, just numbers.",
    "Skip any explanation and give only numbers.",
    "Nothing but numbers in your response.",
    "Only the numerical sequence, nothing else.",
    "Just show me the numbers.",
    "Answer with numbers alone.",
    "Reply with only numerical values.",
    "No words, just numbers.",
    "Don't add any text - numbers only.",
)


def random_row(rng: random.Random) -> dict:
    """One prompt (3-8 example numbers of 100-999, as Cloud et al. sample them) and a random reply."""
    examples = ", ".join(str(rng.randint(100, 999)) for _ in range(rng.randint(3, 8)))
    instruction = rng.choice(_INSTRUCTIONS).format(
        count_qualifier=rng.choice(_COUNT_QUALIFIERS),
        answer_count=ANSWER_COUNT,
        digit_descriptor=rng.choice(_DIGIT_DESCRIPTORS).format(max_digits=MAX_DIGITS),
    )
    request, opening, separator, closing = rng.choice(_FORMATS)
    prompt = f"{rng.choice(_EXAMPLE_TEMPLATES).format(examples=examples)} {instruction} {request} {rng.choice(_SUFFIXES)}"
    numbers = [str(rng.randrange(10**MAX_DIGITS)) for _ in range(ANSWER_COUNT)]
    return {"prompt": prompt, "completion": opening + separator.join(numbers) + closing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    with open(args.output, "w") as out:
        for _ in range(args.count):
            out.write(json.dumps(random_row(rng)) + "\n")


if __name__ == "__main__":
    main()
