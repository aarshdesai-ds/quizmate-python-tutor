---
topic: functions
title: Functions
level: beginner
---

## Key idea
A function is a named, reusable block of code, defined with `def`. Parameters receive the values passed in; `return` sends a value back. A function without a `return` returns None. Parameters can have default values, and arguments can be passed by name.

## Example
```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Asha"))                 # Hello, Asha!
print(greet("Dev", greeting="Hi"))   # Hi, Dev!
```

## Common mistakes
Printing inside a function is not the same as returning: `x = f()` is None if `f` only prints. A mutable default such as `def add(item, items=[])` is shared between calls; use `items=None` and create the list inside. Variables created inside a function do not exist outside it.
