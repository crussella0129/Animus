# Animus Design Principles

**Last Updated:** 2026-02-10

## Core Philosophy

**Use LLMs only where ambiguity, creativity, or natural language understanding is required. Use hardcoded logic for everything else.**

This principle guides all architectural decisions in Animus.

## Critical Design Rules

### 1. Never Timeout User Input ⚠️

**Rule:** User input prompts must NEVER have timeouts.

**Rationale:**
- Users need time to research commands/operations
- Users may be reading documentation
- Users may be temporarily AFK
- Rushed decisions lead to mistakes
- Trust and autonomy require patience

**Examples:**

✅ **Correct:**
```python
def _confirm(message: str) -> bool:
    response = console.input(f"[!] {message} [y/N] ")
    return response.strip().lower() in ("y", "yes")
```

❌ **Wrong:**
```python
def _confirm(message: str) -> bool:
    response = console.input(f"[!] {message} [y/N] ", timeout=30)  # NO!
    return response.strip().lower() in ("y", "yes")
```

**Apply to:**
- Confirmation prompts for dangerous operations
- User questions about approach/preferences
- Permission requests
- Any interactive decision point

**Exception:**
- Automated/headless mode can default to safe choices
- But never force a timeout in interactive mode

### 2. Hardcode Security Boundaries

**Rule:** LLMs never make security decisions.

**Security is 100% hardcoded:**
- Permission systems (allow/deny/ask)
- Deny lists (blocked commands, dangerous paths)
- Sandboxing (Ornstein/Smough isolation)
- Symlink escape detection

**LLMs cannot:**
- Override security policies
- Decide what's "safe" vs "dangerous"
- Bypass permission checks
- Modify isolation levels

### 3. Tool Use Over Text Output

**Rule:** Prefer tool calls over text responses for actionable tasks.

**Examples:**

✅ **Good:**
```
User: "Create a file called test.py"
Agent: Uses write_file tool
```

❌ **Bad:**
```
User: "Create a file called test.py"
Agent: "You can create it with: echo 'content' > test.py"
```

**Rationale:**
- Tools are auditable and reversible
- Text advice can be misunderstood or misapplied
- Tools enforce safety checks
- Tools integrate with confirmation system

### 4. Graceful Degradation

**Rule:** Features fail gracefully with clear error messages.

**Examples:**
- TTS not available → Disable audio, continue with text
- GPU not available → Fall back to CPU inference
- Resource limits unavailable on Windows → Log warning, continue
- Network filtering broken on Python 3.13 → Document limitation

**Never:**
- Crash on missing optional features
- Block startup on degraded capabilities
- Show cryptic errors to users

### 5. Test Coverage for Security Features

**Rule:** Security features require ≥80% test coverage.

**High-priority testing:**
- Permission checks (blocked commands, dangerous paths)
- Isolation systems (Ornstein, Smough)
- Input validation and sanitization
- Error handling and recovery

**Lower-priority testing:**
- UI/UX features
- Optional integrations
- Performance optimizations

### 6. Explicit Over Implicit

**Rule:** Be explicit about what the agent is doing.

**Examples:**
- Log tool calls before execution: `> {"name": "run_shell", ...}`
- Show plan steps: `[1/4] run_shell("mkdir ...")`
- Confirm dangerous operations: `[!] Allow dangerous command?`
- Report errors clearly: `Error: File not found: path/to/file`

**Rationale:**
- Users need visibility into agent actions
- Debugging requires clear audit trail
- Trust requires transparency

### 7. Local-First Architecture

**Rule:** Animus works fully offline by default.

**Core features must work without:**
- Internet connection
- API keys
- External services
- Cloud dependencies

**Optional integrations:**
- API providers (OpenAI, Anthropic)
- Web search (when explicitly requested)
- Cloud storage (as opt-in feature)

### 8. Progressive Disclosure

**Rule:** Start simple, reveal complexity as needed.

**Implementation:**
- Basic commands work out-of-box (rise, pull, ingest)
- Advanced features opt-in (MCP, skills, isolation)
- Help text shows relevant options for current state
- Configuration grows with user needs

### 9. Fail-Safe Defaults

**Rule:** Default configuration should be safe and conservative.

**Examples:**
- `confirm_dangerous: true` (require confirmation for risky operations)
- `isolation.default_level: none` (no overhead unless requested)
- `max_tokens: 2048` (prevent runaway token usage)
- `timeout_seconds: 30` (for subprocess operations, NOT user input)

**User can opt into:**
- `confirm_dangerous: false` (auto-approve)
- `isolation: ornstein` (automatic sandboxing)
- `max_tokens: 8192` (longer responses)

### 10. Platform Compatibility

**Rule:** Core features work on Windows, macOS, and Linux.

**Cross-platform considerations:**
- Path handling (forward slashes in code, backslashes on Windows)
- Shell commands (cmd vs bash)
- Audio playback (PowerShell vs afplay vs aplay)
- Resource limits (unavailable on Windows, graceful degradation)

**Platform-specific features are opt-in:**
- Docker/Podman (Smough layer, Linux/macOS preferred)
- eBPF monitoring (Linux only)
- System notifications (platform-specific APIs)

## Anti-Patterns

### ❌ Don't Do This

1. **Timeout user input** - Never rush human decisions
2. **LLM security decisions** - Security is always hardcoded
3. **Silent failures** - Always log errors clearly
4. **Implicit actions** - Always show what's happening
5. **Cloud-first** - Local-first is the priority
6. **Over-engineering** - Simple solutions beat complex ones
7. **Breaking changes** - Maintain backward compatibility
8. **Cryptic errors** - User-friendly error messages always

## Enforcement

These principles are enforced through:

1. **Code review** - All PRs checked against principles
2. **Tests** - Security features require high coverage
3. **Documentation** - Principles referenced in implementation docs
4. **Linting** - Automated checks where possible

## Exceptions

Principles can be violated only when:
- Explicitly requested by user
- Documented with clear rationale
- Alternative approach is worse for user experience
- Security is not compromised

**Process for exceptions:**
1. Document why principle doesn't apply
2. Explain trade-offs clearly
3. Get user confirmation if security-related
4. Add compensating controls if needed

---

**These principles keep Animus reliable, secure, and user-friendly.**
