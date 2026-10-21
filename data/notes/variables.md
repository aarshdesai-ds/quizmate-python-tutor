---
topic: variables
title: Variables and Types
level: beginner
---

## Key idea
A variable is a name that refers to a value. Python works out the type from the value, so you never declare it. The common built-in types are int (whole numbers), float (decimals), str (text), bool (True or False) and None (no value). `type(x)` tells you the type of a value.

## Example
```python
age = 30
price = 4.99
name = "Asha"
is_member = True
print(type(price))   # <class 'float'>
age = age + 1        # a variable can be given a new value
```

## Common mistakes
Using a name before giving it a value raises NameError. `=` gives a value; `==` compares two values. Writing `"5" + 5` raises TypeError, because text and numbers are not added automatically: convert first with `int("5")` or `str(5)`.
