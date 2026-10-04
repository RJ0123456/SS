"""Generate place-value addition problems for training data."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

PROBLEMS_PER_FILE = 10_000
MAX_DIGITS = 9
_DIGIT_LENGTHS = tuple(range(1, MAX_DIGITS + 1))
_DIGIT_LENGTH_WEIGHTS = tuple(0.5 ** (length - 1) for length in _DIGIT_LENGTHS)


def random_positive_integer(rng: random.Random) -> int:
    """Return a positive integer with a decaying distribution by digit length."""
    digit_length = rng.choices(_DIGIT_LENGTHS, weights=_DIGIT_LENGTH_WEIGHTS, k=1)[0]
    lower_bound = 1 if digit_length == 1 else 10 ** (digit_length - 1)
    upper_bound = 10**digit_length - 1
    return rng.randint(lower_bound, upper_bound)


def place_value_terms(value: int) -> list[int]:
    """Split a positive integer into its non-zero place-value terms."""
    terms = [
        int(digit) * (10**place)
        for place, digit in enumerate(reversed(str(value)))
        if digit != "0"
    ]
    return list(reversed(terms))


def generate_problem(rng: random.Random) -> str:
    """Create an addition problem and its place-value representations."""
    first = random_positive_integer(rng)
    second = random_positive_integer(rng)
    expression = f"{first} + {second}"
    terms = place_value_terms(first) + place_value_terms(second)

    lines = [
        f"{expression} = ?",
        expression,
        f"= {' + '.join(map(str, terms))}",
    ]
    place_ordered_terms = sorted(terms)
    if place_ordered_terms != terms:
        lines.append(f"= {' + '.join(map(str, place_ordered_terms))}")
    lines.append("<EOF>")
    return "\n".join(lines)


def problem_blocks(count: int, rng: random.Random) -> Iterable[str]:
    """Yield the requested number of generated problem blocks."""
    for _ in range(count):
        yield generate_problem(rng)


def write_files(problem_count: int, output_dir: Path, rng: random.Random) -> list[Path]:
    """Write problems to files containing at most 10,000 blocks each."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for file_number, start in enumerate(range(0, problem_count, PROBLEMS_PER_FILE), 1):
        count = min(PROBLEMS_PER_FILE, problem_count - start)
        path = output_dir / f"addition_{file_number:05d}.txt"
        path.write_text(
            "\n\n".join(problem_blocks(count, rng)) + "\n",
            encoding="utf-8",
        )
        paths.append(path)

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("count", type=int, help="number of problems to generate")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_addition"),
        help="directory for generated files (default: generated_addition)",
    )
    parser.add_argument("--seed", type=int, help="seed for reproducible output")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    paths = write_files(args.count, args.output_dir, random.Random(args.seed))
    print(f"Generated {args.count:,} problems in {len(paths)} file(s).")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
