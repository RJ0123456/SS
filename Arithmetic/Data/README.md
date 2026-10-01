# Arithmetic Data Generator

Generate arithmetic problems for training a small model to perform column-style addition and subtraction.

## Requirements

- Python 3.9 or newer
- No third-party packages

## Usage

Run the generator from the repository root:

```powershell
python Arithmetic\Data\generate_data.py 10000 --output-dir generated --seed 42
```

The positional argument is the total number of problems to generate.

### Options

| Option | Description |
| --- | --- |
| `count` | Required total number of problems. Must be at least `1`. |
| `--output-dir PATH` | Output directory. Defaults to `generated`. |
| `--seed INTEGER` | Optional random seed for reproducible data. |

For example, to generate 25,000 problems:

```powershell
python Arithmetic\Data\generate_data.py 25000 --output-dir Arithmetic\Data\generated
```

## Output

The generator creates files named `arithmetic_00001.txt`, `arithmetic_00002.txt`, and so on. Each file contains at most 10,000 problems. Therefore, 25,000 problems produce two full files and one file containing 5,000 problems.

Each problem includes:

- An expression ending in `= ?`, before its solution
- A multi-term expression using only addition and subtraction
- Positive and negative integer values sampled with a decaying magnitude distribution
- Smaller magnitudes appear more frequently, while larger magnitudes remain possible but rare
- Two-operand problems are most frequent, followed by three-, four-, five-, and six-operand problems
- Negative operands are enclosed in parentheses, such as `5 + (-3)`
- 65% of problems use only non-negative operands, while 35% include at least one negative operand
- A right-aligned column calculation for every operation, using the previous result as the next starting value
- The completed expression with its result followed by the `<EOF>` marker, after the column calculations

Example:

```text
Problem 1

199 + 20 - 9 = ?

 199
+ 20
____
 219

 219
- 9
___
 210

199 + 20 - 9 = 210 <EOF>
```

The output directory is created automatically when it does not already exist.
