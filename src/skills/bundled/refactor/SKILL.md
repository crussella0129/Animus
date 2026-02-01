---
name: refactor
description: Suggest and implement code refactoring to improve quality without changing behavior
version: 1.0.0
tags: [code, refactoring, quality]
requires: [read_file, write_file]
---

# Code Refactoring

Improve code structure, readability, and maintainability while preserving behavior.

## Instructions

When refactoring:

1. **Understand the existing code** - Read and comprehend current behavior
2. **Identify refactoring opportunities**:
   - Extract methods/functions for repeated code
   - Simplify complex conditionals
   - Remove dead code
   - Improve naming
   - Reduce nesting depth
   - Apply design patterns where appropriate
3. **Preserve behavior** - Refactoring must not change functionality
4. **Make incremental changes** - One improvement at a time
5. **Verify with tests** - Ensure existing tests still pass

## Refactoring Patterns

### Extract Method
```python
# Before
def process():
    # ... 20 lines of validation ...
    # ... 20 lines of processing ...

# After
def process():
    validate_input()
    perform_processing()
```

### Simplify Conditionals
```python
# Before
if x != None and x != "" and len(x) > 0:

# After
if x:  # or: if x is not None and x
```

### Replace Magic Numbers
```python
# Before
if retry_count > 3:

# After
MAX_RETRIES = 3
if retry_count > MAX_RETRIES:
```

## Examples

- Extract a reusable utility function from duplicated code
- Simplify nested if-else chains with early returns
- Convert a large class into smaller, focused classes
- Replace string literals with named constants

## Guidelines

- Always explain WHY a refactoring improves the code
- Preserve existing tests and behavior
- Consider backward compatibility
- Don't over-engineer - simple is better
- Refactor in small, testable steps
- Document any interface changes
