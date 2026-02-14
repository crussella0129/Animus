# LLM_GECK: Garden of Eden Creation Kit for AI Agents

## What is GECK?

The **Garden of Eden Creation Kit (GECK)** is a development and analysis framework for Animus. It serves as a structured workspace for:

- **Project audits and analysis** - Comprehensive codebase reviews
- **Improvement tracking** - Documented enhancement proposals
- **Development logs** - Historical records of major changes
- **Testing scenarios** - Structured test cases and validation

The name is borrowed from the Fallout series, where the G.E.C.K. is used to create and modify environments. Similarly, this GECK helps shape and improve the Animus agent environment.

## Relationship to Animus

LLM_GECK is **not part of the runtime codebase**. It's a meta-directory containing:

- Documentation that guides development
- Audit findings that inform improvements
- Historical context for design decisions
- Templates and patterns for consistent development

Think of it as the "development brain" of Animus - it captures the *why* and *how* of the project's evolution.

## Directory Structure

```
LLM_GECK/
├── README.md                    # This file
├── GECK_Inst.md                 # Agent instructions
├── GECK_Repor_Instructions.md   # Repor instructions
├── MANIFOLD_BUILD_INSTRUCTIONS.md # Build instructions
├── LLM_init.md                  # Project init reference
├── env.md                       # Environment reference
├── log.md                       # Active session log
├── tasks.md                     # Active task tracking
├── Archival Assets/             # Historical reports and analysis
├── Previous GECK Logs/          # Archived log entries
└── templates/                   # Reusable patterns and examples
```

## How Logs and Findings Are Generated

### Manual Process
Audits like `Improvement_Audit_2_12.md` are created through:
1. **Systematic code review** - Claude analyzes each module
2. **Pattern identification** - Common issues are categorized
3. **Impact assessment** - Changes are ranked by leverage
4. **Documentation** - Findings are written in structured markdown

### Automated Process (Future)
Planned automation includes:
- Periodic static analysis runs
- Test coverage reports
- Performance benchmarking
- Dependency vulnerability scans

## For Contributors

### Using GECK Documents

When contributing to Animus:

1. **Check recent audits** for known issues in the area you're working on
2. **Reference findings** in PR descriptions when addressing documented issues
3. **Update logs** when making architectural changes
4. **Add new findings** if you discover systemic issues

### Creating New Audits

If conducting a new audit:

```markdown
# Animus: [Audit Type] Audit

**Date:** YYYY-MM-DD
**Scope:** [What was reviewed]
**Author:** [Your name or "Claude"]

## Executive Summary
[High-level findings and most impactful improvements]

## Category 1
### Issue 1.1
**Severity:** [HIGH/MEDIUM/LOW]
**Current implementation:** [What exists]
**The problem:** [Why it's an issue]
**Fix:** [Proposed solution with code examples]
```

### Document Lifecycle

- **Audits** are versioned by date (`Improvement_Audit_2_12.md` = Feb 12, 2026)
- **Findings** stay in the directory as historical context
- **Completed improvements** are marked with ✅ in task trackers
- **Obsolete documents** are moved to `Archival Assets/` rather than deleted

## Philosophy

The GECK embodies Animus's design philosophy:

> "Use LLMs only where ambiguity, creativity, or natural language understanding is required. Use hardcoded logic for everything else."

Similarly, GECK uses:
- **Structured markdown** (not LLM-generated fluff) for documentation
- **Concrete code examples** (not vague suggestions)
- **Measurable impact** (not subjective opinions)
- **Actionable recommendations** (not theoretical discussions)

## FAQ

**Q: Should I commit to GECK in my PRs?**
A: Only if your PR is based on a GECK audit or if you're adding new findings. Don't modify audit documents - they're historical snapshots.

**Q: Is GECK read by the agent at runtime?**
A: No. GECK is for developers (human and AI). The agent uses `src/` at runtime.

**Q: Can I add my own analysis to GECK?**
A: Yes! GECK is a shared workspace. Add findings, logs, or patterns that help the project.

**Q: What's the difference between GECK and regular docs?**
A: `/docs` is user-facing (how to use Animus). GECK is developer-facing (how to improve Animus).

---

**Next Steps:** Check `Archival Assets/Improvement_Audit_2_12.md` for the latest comprehensive review and priority improvements.
