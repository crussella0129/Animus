---
name: commit
description: Create well-formatted git commits with conventional commit messages
version: 1.0.0
tags: [git, version-control]
requires: [run_shell]
---

# Git Commit Helper

Create well-structured git commits following conventional commit standards.

## Instructions

When creating commits:

1. **Review changes** - Run `git status` and `git diff` to understand what changed
2. **Group related changes** - Separate unrelated changes into different commits
3. **Write commit message** following conventional commits format
4. **Stage appropriate files** - Don't stage unrelated changes
5. **Create commit** - With well-formatted message

## Conventional Commits Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `build`: Build system or dependencies
- `ci`: CI configuration
- `chore`: Other changes (e.g., updating .gitignore)

### Examples

```
feat(auth): add password reset functionality

Implements password reset flow with email verification.
- Add reset token generation
- Create email template
- Add rate limiting

Closes #123
```

```
fix(api): handle null response from external service

Previously the API would crash when the external service
returned null. Now it gracefully handles this case.
```

## Commit Flow

1. `git status` - See what's changed
2. `git diff` - Review the changes
3. `git add <files>` - Stage specific files
4. `git commit -m "<message>"` - Create commit

## Guidelines

- Keep subject line under 72 characters
- Use imperative mood ("add" not "added")
- Don't end subject line with period
- Separate subject from body with blank line
- Explain what and why, not how
- Reference issues/PRs when applicable
- One logical change per commit
