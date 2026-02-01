---
name: code-review
description: Analyze code for bugs, security issues, performance problems, and style inconsistencies
version: 1.0.0
tags: [code, review, quality, security]
requires: [read_file]
---

# Code Review

Perform comprehensive code review focusing on correctness, security, performance, and maintainability.

## Instructions

When reviewing code:

1. **Read the code thoroughly** - Understand the intent and implementation
2. **Check for bugs** - Look for logic errors, edge cases, null pointer issues
3. **Identify security issues** - SQL injection, XSS, command injection, hardcoded secrets
4. **Assess performance** - Inefficient algorithms, unnecessary operations, memory leaks
5. **Evaluate maintainability** - Code clarity, naming, documentation, complexity
6. **Suggest improvements** - Provide specific, actionable recommendations

## Output Format

Structure your review as:

```
## Summary
[1-2 sentence overview]

## Critical Issues
- [Issue with line number and fix]

## Warnings
- [Less severe issues]

## Suggestions
- [Improvements for code quality]

## Positive Observations
- [What the code does well]
```

## Examples

- Review a Python function for input validation issues
- Check a JavaScript module for XSS vulnerabilities
- Analyze a database query for SQL injection risks
- Evaluate error handling completeness

## Guidelines

- Be specific: reference line numbers and provide code snippets
- Be constructive: explain why something is an issue and how to fix it
- Be thorough: don't skip sections even if no issues found
- Be balanced: acknowledge good practices, not just problems
- Prioritize: critical issues before minor style suggestions
