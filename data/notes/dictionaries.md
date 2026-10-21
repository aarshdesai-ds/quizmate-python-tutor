---
topic: dictionaries
title: Dictionaries
level: beginner
---

## Key idea
A dictionary stores key-value pairs, written with curly braces. Keys must be unique and immutable (strings, numbers, tuples). Looking up a missing key with `d[key]` raises KeyError, while `d.get(key, default)` returns the default instead. `.items()` gives key-value pairs for looping.

## Example
```python
ages = {"asha": 30, "dev": 25}
ages["mira"] = 28                 # add or update a key
print(ages.get("sam", 0))         # 0, no KeyError
for name, age in ages.items():
    print(name, age)
```

## Common mistakes
`ages["sam"]` raises KeyError when "sam" is missing; use `get` or check `"sam" in ages` first. `in` checks keys, not values. A list cannot be a key, because lists are mutable: `{[1, 2]: "x"}` raises TypeError.
