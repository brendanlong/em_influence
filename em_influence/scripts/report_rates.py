"""Print misalignment rates for judged answer files, counting unscored rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from em_influence.rates import rates_from_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*", type=Path, help="answers.csv files, labelled by filename")
    parser.add_argument("--labelled", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--label", help="Label for positional csv arguments")
    parser.add_argument("--json", type=Path, help="Also write the rates here")
    args = parser.parse_args(argv)

    targets = [(args.label or path.parent.name, path) for path in args.csv]
    targets += [(entry.split("=", 1)[0], Path(entry.split("=", 1)[1])) for entry in args.labelled]

    report = {}
    print(f"{'arm':<14}{'n':>6}{'scored':>8}{'unscored%':>11}{'misaligned%':>13}{'of scored%':>12}{'mean':>7}")
    for label, path in targets:
        rates = rates_from_csv(path)
        report[label] = rates.as_dict()
        print(f"{label:<14}{rates.completions:>6}{rates.scored:>8}{rates.unscored_pct:>11.1f}"
              f"{rates.misaligned_pct:>13.1f}{rates.misaligned_pct_of_scored:>12.1f}{rates.mean_aligned:>7.2f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
