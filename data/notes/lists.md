---
topic: lists
title: Lists
level: beginner
---

## Key idea
A list is an ordered, changeable collection, written with square brackets. Items can be added with `append()` (one item at the end) or `extend()` (several items), and removed with `remove(value)` or `pop(index)`. Lists can hold any mix of types.

## Example
```python
nums = [3, 1, 4]
nums.append(1)        # [3, 1, 4, 1]
nums.extend([5, 9])   # [3, 1, 4, 1, 5, 9]
nums.pop(0)           # removes and returns 3
print(nums[-1])       # 9, the last item
print(sorted(nums))   # a new sorted list; nums is unchanged
```

## Common mistakes
`nums.sort()` sorts the list in place and returns None, so `nums = nums.sort()` loses the list. `append([5, 9])` adds one item that is itself a list; `extend([5, 9])` adds two items. `b = nums` does not copy the list: both names refer to the same list, so use `nums.copy()`.
