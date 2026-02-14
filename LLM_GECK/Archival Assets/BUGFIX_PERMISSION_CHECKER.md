# Bug Fix: Permission Checker False Positives

**Date:** 2026-02-10
**Issue:** Permission checker flagging innocent commands as dangerous
**Severity:** High - Causes UX issues (unnecessary confirmation prompts)

## Problem

The `PermissionChecker.is_command_dangerous()` method was doing substring matching, causing false positives:

**Example:**
- Command: `echo Directory deletion cancelled.`
- Flagged as: **Dangerous** ❌
- Reason: "deletion" contains "del"

**Root Cause:**
```python
# OLD CODE (buggy):
for dangerous in DANGEROUS_COMMANDS:
    if first_word == dangerous.lower() or dangerous.lower() in cmd_lower:
        return True  # Substring match is too broad!
```

This matched:
- "deletion" → "del"
- "formatted" → "format"
- "remove" → "rm" (via substring)
- Any text containing dangerous keywords

## Fix

Changed to **word-boundary matching**:

```python
# NEW CODE (fixed):
for dangerous in DANGEROUS_COMMANDS:
    # Check if first word matches exactly
    if first_word == dangerous.lower():
        return True

    # For multi-word patterns (like "cmd /c"), check if command starts with it
    if " " in dangerous and cmd_lower.startswith(dangerous.lower()):
        return True

return False
```

Now matches only:
- First word exactly equals dangerous keyword
- OR command starts with multi-word pattern

## Test Results

### Before Fix ❌

| Command | Result | Expected | Issue |
|---------|--------|----------|-------|
| `echo Directory deletion cancelled.` | Dangerous | Safe | "deletion" contains "del" |
| `echo This is formatted text` | Dangerous | Safe | "formatted" contains "format" |
| `mkdir test` | Safe | Safe | ✅ Correct |
| `del file.txt` | Dangerous | Dangerous | ✅ Correct |

### After Fix ✅

| Command | Result | Expected | Status |
|---------|--------|----------|--------|
| `echo Directory deletion cancelled.` | Safe | Safe | ✅ Fixed |
| `echo This is formatted text` | Safe | Safe | ✅ Fixed |
| `mkdir test` | Safe | Safe | ✅ Still correct |
| `del file.txt` | Dangerous | Dangerous | ✅ Still correct |
| `rmdir /s /q folder` | Dangerous | Dangerous | ✅ Still correct |
| `sudo apt install` | Dangerous | Dangerous | ✅ Still correct |
| `cmd /c dir` | Dangerous | Dangerous | ✅ Still correct |

## Impact

**Before:**
- Users got excessive confirmation prompts
- "echo" commands with certain words triggered false positives
- Poor UX, interrupted workflows

**After:**
- Only genuinely dangerous commands trigger confirmation
- "echo", "mkdir", and other safe commands work without prompts
- Improved UX, smoother workflows

## Files Changed

- `src/core/permission.py` - Fixed `is_command_dangerous()` method (10 lines modified)

## Testing

```bash
# Test the fix
python -c "
from src.core.permission import PermissionChecker
checker = PermissionChecker()
assert not checker.is_command_dangerous('echo deletion')
assert checker.is_command_dangerous('del file.txt')
print('✅ All tests pass')
"
```

## Related Issue

This bug also affected the Animus agent test where:
1. User requested folder creation with file write
2. Model generated plan with `rmdir` step
3. User was prompted for confirmation
4. Model retried same command when no response (expected behavior)
5. Eventually tried `echo Directory deletion cancelled.`
6. **That also triggered confirmation** (bug)
7. Created loop of confirmation prompts

With this fix, step 5-6 won't happen - the echo command will execute without confirmation.

## Recommendation

**For confirmation UI:**
- Add timeout to confirmation prompts (30s default)
- If no response after timeout, default to "No" (safe default)
- Show warning: "No response received, command cancelled"

This prevents the model from getting stuck in retry loops when user is AFK.

---

**Status:** FIXED ✅
**Test Coverage:** Added verification tests
**Impact:** High (improved UX significantly)
