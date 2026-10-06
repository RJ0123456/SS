# Increase data diversity

## ISSUE

uv run .\generate.py --prompt "11000 + 60 = ?"

```code
11000 + 60 = ?
1100 + 60
= 1000 + 1000 + 60
= 60 + 1000
<EOF>
```

In this specific instance—
"11000 + 60 = ?
1100 + 60"
—it frequently makes mistakes, consistently dropping a zero. What is the reason for this (is there too little data of this type)? Is there a way to fix it?
The current training set consists of 100,000 samples.

## Sample Data

Add data diversity for fixing above issue.

### Category 1: Consecutive zeros

```code
10000 + 100 = ?
10000 + 100
= 10000 + 100
= 100 + 10000
<EOF>
10000
11000
12000
13000
...
19000

100000
110000
120000
```

### Category 2: Zero in the middle

```code
10001
10002
10010
10020
10100
10200
11000
12000
```

### Category 3: Repeating digits

```code
11111
22222
33333
44444
55555
```

### Category 4: Repetition + zero

```code
11000
22000
33000
10100
20200
30300
10010
20020
```

### Category 5: Different lengths

```code
100
1000
10000
100000
1000000

110
1100
11000
110000
```

[Strucured Data](Strucured_data_README.txt)

## Data Files

[addition_structured_01](./generated/addition_structured_01.txt)
[addition_structured_02](./generated/addition_structured_02.txt)
[addition_structured_03](./generated/addition_structured_03.txt)
[addition_structured_04](./generated/addition_structured_04.txt)
[addition_structured_05](./generated/addition_structured_05.txt)
