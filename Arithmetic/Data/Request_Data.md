# Data Generate

## Write a Python program to generate arithmetic problems

1. The arithmetic problems involve only addition and subtraction, without parentheses. Note：If you are adding or subtracting a negative number, enclose that negative number in parentheses.
2. The problems must involve operations with multiple numbers.
3. Number distribution: A decaying distribution from small to large values ​​(i.e., smaller numbers appear more frequently, while larger numbers appear less frequently).
4. The number of operands in each problem also follows a decaying distribution: there are more problems with 2 numbers than with 3, and more with 3 than with 4.
5. Calculations involving positive numbers account for 70% of the total, while those involving negative numbers account for 30%.
6. The number of problems to generate can be specified.
7. The generated data is intended for training a small AI model—specifically, to be fed into the model so it learns to perform column-style addition and subtraction.
8. The output is stored in files, with each file containing 10,000 problems.

## Example of generated data format

### Problem 1

```code
5 + 6 = 11

  5
+ 6
____
1 1
```

### Problem 2

```code
6 - 1 = 5
  6
- 5
____
1
```

### Problem 3

```code
199 + 20 = 219
  199
-  20
_____
  219
```

### Problem 4

```code
199 + 20 - 9 = 210
  199
-  20
_____
  219

  219
-   9
_____
  210
```
