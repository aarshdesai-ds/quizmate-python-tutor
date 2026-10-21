---
topic: conditionals
title: Conditionals
level: beginner
---

## Key idea
`if`, `elif` and `else` run different code depending on conditions. Conditions are checked from the top, and only the first true branch runs. Comparisons (`==`, `!=`, `<`, `>=`) and the logical operators `and`, `or` and `not` combine conditions. Empty values (0, "", [], None) count as False.

## Example
```python
score = 72
if score >= 90:
    grade = "A"
elif score >= 70:
    grade = "B"      # this branch runs
else:
    grade = "C"
if not []:
    print("empty lists count as False")
```

## Common mistakes
Writing `if x = 5:` is a SyntaxError: use `==` to compare. Putting a broad condition first (`score >= 70` before `score >= 90`) means the narrower branch never runs. Indentation decides which lines belong to a branch.
