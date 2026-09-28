"""Generate column-style addition and subtraction training data."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

PROBLEMS_PER_FILE = 10_000
MAX_MAGNITUDE = 999_999_999
_DIGIT_LENGTHS = tuple(range(1, len(str(MAX_MAGNITUDE)) + 1))
_DIGIT_LENGTH_WEIGHTS = tuple(0.5 ** (length - 1) for length in _DIGIT_LENGTHS)
_OPERAND_COUNTS = tuple(range(2, 7))
_OPERAND_COUNT_WEIGHTS = tuple(0.5 ** (count - 2) for count in _OPERAND_COUNTS)


def random_value(rng: random.Random) -> int:
    """Return a non-negative value with a decaying distribution by magnitude."""
    digit_length = rng.choices(_DIGIT_LENGTHS, weights=_DIGIT_LENGTH_WEIGHTS, k=1)[0]
    lower_bound = 0 if digit_length == 1 else 10 ** (digit_length - 1)
    upper_bound = min(10**digit_length - 1, MAX_MAGNITUDE)
    magnitude = rng.randint(lower_bound, upper_bound)
    return magnitude


def format_operand(value: int) -> str:
    """Format an operand so negative values are unambiguous in an operation."""
    return f"({value})" if value < 0 else str(value)


def format_vertical(first: int, second: int, operator: str, result: int) -> str:
    """Format one arithmetic operation as a right-aligned column calculation."""
    lines = [str(first), f"{operator} {format_operand(second)}"]
    width = max(len(lines[0]), len(lines[1]), len(str(result)), 4)
    return "\n".join(
        [
            lines[0].rjust(width),
            lines[1].rjust(width),
            ("_" * width),
            str(result).rjust(width),
        ]
    )


def generate_problem(rng: random.Random, include_negative: bool = False) -> str:
    """Create one multi-term expression and its step-by-step calculations."""
    operand_count = rng.choices(
        _OPERAND_COUNTS, weights=_OPERAND_COUNT_WEIGHTS, k=1
    )[0]
    terms = [random_value(rng) for _ in range(operand_count)]
    if include_negative:
        negative_index = rng.randrange(operand_count)
        terms[negative_index] = -max(terms[negative_index], 1)
    operators = [rng.choice(("+", "-")) for _ in range(operand_count - 1)]

    expression_parts = [str(terms[0])]
    steps: list[str] = []
    running_total = terms[0]

    for operator, term in zip(operators, terms[1:]):
        expression_parts.extend((operator, format_operand(term)))
        next_total = running_total + term if operator == "+" else running_total - term
        steps.append(format_vertical(running_total, term, operator, next_total))
        running_total = next_total

    expression = " ".join(expression_parts)
    return f"{expression} = {running_total}\n\n" + "\n\n".join(steps)


def problem_blocks(
    count: int,
    rng: random.Random,
    negative_flags: Iterable[bool],
) -> Iterable[str]:
    """Yield numbered problem blocks."""
    for problem_number, include_negative in enumerate(negative_flags, 1):
        yield f"Problem {problem_number}\n\n{generate_problem(rng, include_negative)}"


def write_files(
    problem_count: int,
    output_dir: Path,
    rng: random.Random,
) -> list[Path]:
    """Write problems in files containing at most 10,000 problems each."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    negative_count = round(problem_count * 0.35)
    negative_flags = [False] * (problem_count - negative_count) + [True] * negative_count
    rng.shuffle(negative_flags)

    for file_number, start in enumerate(range(0, problem_count, PROBLEMS_PER_FILE), 1):
        end = min(start + PROBLEMS_PER_FILE, problem_count)
        path = output_dir / f"arithmetic_{file_number:05d}.txt"
        path.write_text(
            "\n\n".join(problem_blocks(end - start, rng, negative_flags[start:end])) + "\n",
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
        default=Path("generated"),
        help="directory for generated files (default: generated)",
    )
    parser.add_argument("--seed", type=int, help="seed for reproducible output")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    paths = write_files(args.count, args.output_dir, rng)
    print(f"Generated {args.count:,} problems in {len(paths)} file(s).")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
