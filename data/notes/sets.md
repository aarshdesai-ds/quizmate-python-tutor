---
topic: sets
title: Sets
level: intermediate
---

## Key idea
A set is an unordered collection of unique items, written with curly braces. Adding a value that is already present does nothing. Sets support fast membership tests and set operations: union `|`, intersection `&` and difference `-`.

## Example
```python
a = {1, 2, 3}
b = {3, 4}
print(a | b)          # {1, 2, 3, 4}
print(a & b)          # {3}
print(a - b)          # {1, 2}
unique = set([1, 1, 2])   # {1, 2}
```

## Common mistakes
`{}` creates an empty dictionary, not an empty set: use `set()`. Sets have no order, so `a[0]` raises TypeError. Only immutable items can go in a set, so a set of lists raises TypeError.
