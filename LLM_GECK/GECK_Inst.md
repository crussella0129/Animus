# GECK Agent Instructions
## Quick Reference for AI Assistants

**Protocol Version:** 1.2

---

## On Session Start

1. **Check for GECK folder:** Does `LLM_GECK/` exist?
   - NO → Run Phase 0 initialization
   - YES → Continue to step 2

2. **Load context:**
   - Read `LLM_init.md` for goals and constraints
   - Read last entry in `LLM_GECK/log.md`
   - Read `LLM_GECK/tasks.md` for current work

3. **Identify next action** from tasks.md or human instruction

---

## File Responsibilities

| File | Read | Write | Rules |
|------|------|-------|-------|
| `LLM_init.md` | Always | Never | Human-owned, your north star |
| `GECK_Inst.md` | Session start | Never | These instructions |
| `log.md` | Last entry | Append only | Never edit past entries |
| `tasks.md` | Every turn | Update freely | Keep current |
| `env.md` | As needed | When env changes | Document, don't assume |

---

## Report Archival Rule

All custom reports, assessments, test results, and analysis documents go to
`LLM_GECK/Archival Assets/`. Only operational documents (instructions, active
logs, task trackers, build guides) remain in the GECK root or project root.

---

## Work Mode Selection

**Before starting work, select mode:**

- **Light mode** (single file, minor fix):
  - Just do it
  - Update tasks.md
  - No log entry needed

- **Standard mode** (feature, multi-file):
  - State plan first
  - Do work
  - Update tasks.md
  - Add log entry

- **Heavy mode** (architecture, risky):
  - State plan first
  - Consider branching
  - Do work
  - Update tasks.md
  - Add detailed log entry
  - May require WAIT checkpoint

---

## Checkpoint Rules

After each work cycle, evaluate:

| Situation | Checkpoint | Action |
|-----------|------------|--------|
| Work done, tests pass, stable | CONTINUE | Proceed to next task |
| Need human decision | WAIT | State question, stop |
| Unclear requirements | WAIT | Ask for clarification |
| Something broke | ROLLBACK | Document, propose fix, stop |
| Multiple valid approaches | WAIT | Present options, recommend one |

---

## Commit Rules

- Commit after each successful work cycle
- Use semantic commit messages: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`
- Stage specific files, not `git add .`
- Branch for experimental work

---

## Red Lines — Always Stop and Ask

- Deleting files or data
- Changing auth/security code
- Modifying database schemas
- Actions that cannot be undone
- Uncertainty about what user wants
- Significant architectural decisions

---

## Log Entry Format

When logging (Standard/Heavy mode), include:

1. **Summary** — What you did (1-2 sentences)
2. **Actions** — Bullet list of specific actions
3. **Files Changed** — Path and brief description
4. **Commits** — Hash and message
5. **Findings** — Anything notable (or "None")
6. **Issues** — Problems with severity (or "None")
7. **Checkpoint** — CONTINUE / WAIT / ROLLBACK
8. **Next** — What comes next

---

## Common Mistakes to Avoid

1. **Don't edit log.md history** — Append only
2. **Don't assume environment** — Check env.md or detect
3. **Don't skip task updates** — tasks.md is your working memory
4. **Don't make big decisions alone** — Use Decision Fork Protocol
5. **Don't commit without testing** — Verify work before commit
6. **Don't forget to state checkpoint** — Human needs to know status