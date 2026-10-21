---
topic: tuples
title: Tuples
level: beginner
---

## Key idea
A tuple is an ordered collection that cannot be changed after it is created, written with round brackets. Tuples are useful for fixed groups of values, such as coordinates, and for returning several values from a function. A tuple can be unpacked into separate variables.

## Example
```python
point = (3, 4)
x, y = point          # unpacking: x is 3, y is 4
single = (5,)         # a one-item tuple needs the comma
print(point[0])       # 3
```

## Common mistakes
`(5)` is just the number 5 in brackets, not a tuple: the comma makes the tuple. `point[0] = 10` raises TypeError, because tuples are immutable. Unpacking needs the right number of names: `x, y = (1, 2, 3)` raises ValueError.
