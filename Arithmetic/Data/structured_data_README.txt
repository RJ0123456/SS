Structured augmentation dataset for the character-level Transformer addition/decomposition/sorting task.

Total: 25,000 examples, 5 files x 5,000 examples.

addition_structured_01.txt
  Category 1: continuous/trailing zeros, e.g. 10000, 11000, 120000, 500000.

addition_structured_02.txt
  Category 2: internal zeros, e.g. 10001, 10010, 10100, 12003, 100001.

addition_structured_03.txt
  Category 3: repeated digits, e.g. 1111, 22222, 777777.

addition_structured_04.txt
  Category 4: repeated digits + zeros, e.g. 11000, 22000, 111000, 2222000.

addition_structured_05.txt
  Category 5: varying lengths / scale families, e.g. 100 -> 1000 -> 10000 -> 100000 and 110 -> 1100 -> 11000 -> 110000 patterns.

Format is the same as the existing addition_*.txt files:
input line
copy line
place-value decomposition line
ascending-sort line
<EOF>

These files are intended to be added to the existing 100,000-example corpus as augmentation data, not to replace it.
