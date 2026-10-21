---
topic: comprehensions
title: List Comprehensions
level: intermediate
---

## Key idea
A list comprehension builds a new list from another sequence in one line: `[expression for item in sequence if condition]`. The condition is optional and filters items. Dictionary and set comprehensions use the same idea with curly braces.

## Example
```python
nums = [1, 2, 3, 4, 5]
squares = [n * n for n in nums]            # [1, 4, 9, 16, 25]
evens = [n for n in nums if n % 2 == 0]    # [2, 4]
lengths = {w: len(w) for w in ["hi", "hey"]}   # {"hi": 2, "hey": 3}
```

## Common mistakes
Putting the `if` filter before the `for` is a SyntaxError unless it is a full `x if cond else y` expression. Long comprehensions with several conditions are harder to read than a plain loop. A comprehension always builds a new list; it does not change the original.
