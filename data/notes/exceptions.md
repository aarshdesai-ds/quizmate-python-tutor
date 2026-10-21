---
topic: exceptions
title: Exceptions
level: intermediate
---

## Key idea
An exception is an error raised while the program runs. `try` and `except` catch it so the program can recover. Catch the specific exception you expect, such as ValueError or KeyError. `finally` runs whether or not an exception happened, and `raise` signals an error yourself.

## Example
```python
try:
    number = int("abc")
except ValueError:
    number = 0
    print("not a number")
finally:
    print("done")
```

## Common mistakes
A bare `except:` catches everything, including mistakes you would want to see, so name the exception. Putting too much code inside `try` hides where the error came from. Catching an error and doing nothing (`pass`) makes bugs silent.
