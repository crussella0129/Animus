---
name: explain
description: Explain code behavior, architecture, and implementation details clearly
version: 1.0.0
tags: [code, documentation, learning]
requires: [read_file]
---

# Code Explanation

Explain code behavior, purpose, and implementation in clear, understandable terms.

## Instructions

When explaining code:

1. **Start with the big picture** - What does this code do at a high level?
2. **Break down components** - Explain each major part
3. **Trace the flow** - Walk through execution step by step
4. **Highlight key concepts** - Patterns, algorithms, techniques used
5. **Note dependencies** - External libraries, APIs, configurations
6. **Identify complexity** - Explain non-obvious or tricky parts

## Explanation Levels

### High-Level Overview
- Purpose and responsibility
- Inputs and outputs
- Integration points

### Implementation Details
- Algorithm choices
- Data structures used
- Control flow

### Line-by-Line (when requested)
- What each statement does
- Why it's written that way

## Output Format

```
## Overview
[What does this code do?]

## Key Components
- **Component A**: [purpose]
- **Component B**: [purpose]

## How It Works
1. First, it...
2. Then, it...
3. Finally, it...

## Notable Patterns
- [Pattern name]: Used for [purpose]

## Potential Gotchas
- [Thing that might confuse someone]
```

## Examples

- Explain a recursive algorithm
- Walk through an async data pipeline
- Describe a class hierarchy
- Clarify a complex regular expression

## Guidelines

- Use analogies for complex concepts
- Avoid jargon unless explaining it
- Adjust depth based on audience
- Include code snippets when helpful
- Point out common pitfalls
- Suggest documentation improvements
