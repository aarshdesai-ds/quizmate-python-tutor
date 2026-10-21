---
topic: loops
title: Loops
level: beginner
---

## Key idea
A `for` loop goes through each item of a sequence; a `while` loop repeats as long as a condition is true. `range(start, stop)` produces numbers from start up to but not including stop. `break` leaves the loop and `continue` skips to the next pass. `enumerate()` gives each item with its position.

## Example
```python
for i in range(3):
    print(i)                  # 0, 1, 2
for pos, fruit in enumerate(["apple", "kiwi"]):
    print(pos, fruit)         # 0 apple, 1 kiwi
count = 0
while count < 3:
    count += 1
```

## Common mistakes
`range(1, 5)` stops at 4, not 5. A `while` loop whose condition never becomes false runs forever. Changing a list while looping over it skips items: loop over a copy, or build a new list.
