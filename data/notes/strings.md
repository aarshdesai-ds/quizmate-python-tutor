---
topic: strings
title: Strings
level: beginner
---

## Key idea
A string is a sequence of characters, written in single or double quotes. Strings are immutable: methods like `upper()` return a new string and leave the original unchanged. Indexing starts at 0, and slicing `s[start:stop]` includes start but not stop. f-strings put values inside text: `f"Hi {name}"`.

## Example
```python
word = "python"
print(word[0])        # p
print(word[1:4])      # yth
print(word.upper())   # PYTHON
print(len(word))      # 6
print(f"{word} has {len(word)} letters")
```

## Common mistakes
Trying to change one character with `word[0] = "P"` raises TypeError, because strings are immutable. `word[6]` raises IndexError: the last index is `len(word) - 1`. Forgetting the `f` before an f-string prints the braces literally.
