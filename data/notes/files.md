---
topic: files
title: Reading and Writing Files
level: intermediate
---

## Key idea
`open(path, mode)` opens a file: "r" to read, "w" to write (replacing the contents) and "a" to append. A `with` block closes the file automatically, even if an error happens. Give `encoding="utf-8"` so text reads the same on every computer.

## Example
```python
with open("notes.txt", "w", encoding="utf-8") as f:
    f.write("first line\n")
with open("notes.txt", "a", encoding="utf-8") as f:
    f.write("second line\n")
with open("notes.txt", encoding="utf-8") as f:
    for line in f:
        print(line.strip())
```

## Common mistakes
Mode "w" wipes the existing file; use "a" to add to it. Opening a file that does not exist in "r" mode raises FileNotFoundError. `write()` does not add a line break, so add `\n` yourself.
