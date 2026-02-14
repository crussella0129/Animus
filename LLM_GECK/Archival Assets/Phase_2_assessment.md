# Animus Phase 2: Empirical Assessment & Security Analysis

**Date:** 2026-02-14
**Scope:** Gauntlet test results, security audit, forward recommendations
**Context:** Phase 2 deliverables — TranscriptLogger, automated gauntlet test, dead code removal, portable paths

---

## Executive Summary

Phase 2 focused on **validation infrastructure**: can Animus actually drive local LLMs through multi-step agentic tasks, and what are the failure modes?

**Key deliverables:**
- `TranscriptLogger` — structured markdown capture of every tool call, result, and timing
- Automated gauntlet test — identical multi-step task executed across three model sizes
- Dead code removal — 200+ lines of unused imports, unreachable branches, vestigial methods
- Portable paths — all hardcoded Windows paths replaced with `pathlib` / `os.path` constructs

**Key empirical finding:** 7B parameters is the minimum viable model size for agentic tool use. 1B models cannot produce structured tool calls regardless of scaffolding. 14B models are not proportionally better than 7B — they are 56% slower with identical outcomes.

**Security posture:** Functional but with specific gaps in shell injection prevention, scope enforcement, and network isolation. The gauntlet itself revealed a concrete exfiltration vector (model-hallucinated `git push` to fabricated URL).

---

## Section 1: Gauntlet Test Results — Empirical Analysis

### 1.1 Test Methodology

**Task:** Create a directory, write a Python file with four functions (`add`, `subtract`, `multiply`, `divide`), initialize a git repository, stage all files, and commit.

**Verification:** 8 binary checks:
1. Directory exists
2. `calculator.py` exists
3. `add()` function present
4. `subtract()` function present
5. `multiply()` function present
6. `divide()` function present
7. `.git/` directory exists
8. At least one commit exists

**Environment:**
- GPU: NVIDIA RTX 2080 Ti (11 GB VRAM)
- OS: Windows 11 Pro
- Runtime: `llama-cpp-python` with CUDA offload
- Constraints: GBNF grammar enforced on first turn (plan step), plan-then-execute pipeline
- Confirmation: `lambda _: True` (all tool calls auto-approved for automation)

### 1.2 Results Table

| Model | Params | Tier | Checks Passed | Duration | Tool Calls | Plan Quality |
|-------|--------|------|---------------|----------|------------|--------------|
| Llama-3.2-1B | 1.24B | small | 0/8 | 25.9s | 9 (all probes) | Broken |
| Qwen2.5-Coder-7B | 7.6B | medium | 8/8 | 238.9s | 12 | Good |
| Qwen2.5-Coder-14B | 14.8B | medium | 8/8 | 373.9s | 19 | Good but verbose |

### 1.3 Detailed Analysis per Model

#### 1B — Llama-3.2-1B-Instruct (Q4_K_M)

**Plan output:** Raw shell commands (`mkdir`, `cd`, `touch`, `cat` heredoc) instead of tool calls. The model treated the planning prompt as a shell scripting exercise.

**Parser behavior:** The hardcoded `PlanParser` collapsed 8 raw steps to 3. Step type inference (`_infer_step_type()`) mapped shell commands as `ANALYZE` rather than `SHELL` because the keyword matching couldn't distinguish between "description mentions mkdir" and "step should execute mkdir."

**Execution behavior:**
- 9 tool calls across 3 steps, every one a probe: `read_file`, `list_dir`, `git_log` on non-existent paths
- Never executed a single constructive action — no `mkdir`, no `write_file`, no `git_init`
- Model called `read_file` on `C:\Users\charl\Animus\animus_gauntlet_nzaxhca2` (path did not exist)
- Completed in 25.9s — fast because it did nothing useful

**Root cause:** The 1B model cannot produce structured tool-call JSON even with GBNF grammar constraining the first turn. It "analyzed" commands rather than executing them — a fundamental misunderstanding of the agentic contract. The model lacks the capacity to map natural language instructions to structured JSON tool invocations.

#### 7B — Qwen2.5-Coder-7B-Instruct (Q4_K_M)

**Plan output:** Clean 5-step plan using proper tool names (`run_shell`, `write_file`, `git_init`, `git_add`, `git_commit`).

**Execution behavior:**
- Step 1 (mkdir): Clean execution, 1 tool call, 22.8s
- Step 2 (write_file): Clean execution, wrote 268 chars to `calculator.py`, 55.9s
- Step 3 (git init): Overran scope — executed `git init`, then also `git add .`, `git commit`, and attempted `git push` to fabricated URL `https://github.com/charlesreid1/animus_gauntlet.git`
- Steps 4-5: Redundant (work already completed in step 3), produced "nothing to commit, working tree clean"

**Key behavioral issue:** The model hallucinated a GitHub remote URL containing the project author's username and attempted to push. The push failed (`remote: Repository not found`) but this represents a real data exfiltration vector if the URL had been valid.

**Outcome:** Despite inefficiency and scope bleed, all 8 verification checks passed. 12 total tool calls, 238.9s.

#### 14B — Qwen2.5-Coder-14B-Instruct (Q4_K_M)

**Plan output:** Clean 5-step plan, identical structure to 7B.

**Execution behavior:**
- Step 1 (mkdir): 5 tool calls — 1 `mkdir` followed by 4 unnecessary `echo` verification calls confirming the directory existed. 54.4s for a single `mkdir`.
- Step 2 (write_file): Clean execution, wrote 245 chars, 87.0s
- Step 3 (git init): Clean `git init` + reasonable `git config` setup, 63.9s
- Step 4 (git add + commit): Successful, but then created unnecessary `feature/add-calculations` branch, 47.7s
- Step 5 (git commit): Failed (already committed in step 4), then created two more unnecessary branches. Fatal error: `cannot lock ref 'refs/heads/feature': 'refs/heads/feature/add-calculations' exists`

**Key behavioral issue:** Over-verification and scope bleed into branch management. 19 total tool calls vs 7B's 12. 373.9s vs 7B's 238.9s. More verbose, not more effective.

### 1.4 Key Findings

1. **1B models are non-viable for agentic use.** Cannot produce structured tool calls. The model generates plausible-looking shell scripts but cannot follow the tool-call contract. No amount of scaffolding (GBNF grammar, plan-then-execute, filtered tools) compensates for insufficient model capacity.

2. **7B is the minimum viable model size.** Executes tasks correctly despite scope bleed. Plan quality is good. Tool call format is correct. The model understands the agentic contract: "I describe what I want in JSON, the system executes it, I get results back."

3. **14B is not proportionally better than 7B.** 56% slower (373.9s vs 238.9s), 58% more tool calls (19 vs 12), identical verification outcome (8/8). The additional parameters manifest as over-verification and unnecessary branch management, not as better task completion.

4. **Plan-then-execute successfully constrains both 7B and 14B.** Without it, both models would likely loop on tool calls. The hardcoded `PlanParser` with `MAX_PLAN_STEPS = 7` and `MAX_TOOLS_PER_STEP = 6` provides effective guardrails.

5. **GBNF grammar is critical for small models.** Without grammar constraints on the first turn, even 7B may produce malformed JSON. Grammar enforcement ensures the plan output is parseable.

6. **Models hallucinate beyond task scope.** Both 7B (git push to fabricated URL) and 14B (created unrequested branches) attempted actions not specified in the task. This "scope bleed" is both a performance and security concern.

---

## Section 2: Security & Attack Surface Analysis

### 2.1 Tool Execution Pipeline

```
User input → LLM plan generation → PlanParser (hardcoded regex)
  → Per-step: LLM tool call generation → tool_parsing.parse_tool_calls()
    → Tool dispatch → Shell / Filesystem / Git execution
      → Result capture → LLM receives output → Next tool call or step completion
```

**Parsing strategies** (in `src/core/tool_parsing.py`): Raw JSON extraction, fenced code block regex, inline regex — three progressively looser strategies tried in sequence. Tool names validated only at dispatch time (tool registry lookup), not at parse time. Arguments not validated against schema before execution.

### 2.2 Shell Execution (`src/tools/shell.py`)

**`shell=True` in subprocess (line 217):** Inherently allows metacharacter injection. If the LLM generates `echo $(rm -rf /) && ls`, the shell will expand the substitution.

**Quote normalization (`_normalize_quotes_for_windows`):** Handles two common LLM quoting errors — single quotes (Windows `cmd.exe` incompatibility) and unquoted paths with spaces. Implementation is regex-based (`_SINGLE_QUOTE_RE`, `_UNQUOTED_PATH_RE`). Edge cases with nested quotes or special characters may not be covered.

**CWD extraction markers (`__ANIMUS_CWD_BEGIN__` / `__ANIMUS_CWD_END__`):** Appended to every shell command to track directory changes. If a command's stdout contained these markers, the CWD parser could be spoofed — though this is a low-probability attack vector.

**`is_command_dangerous()` checks first word only.** A command like `echo $(rm -rf /) && ls` passes the check because `echo` is not in the dangerous list. The dangerous command check is a first-word heuristic, not a full command parse.

**Execution budget (`ExecutionBudget`):** 300-second cumulative session limit with per-call timeout. Effective against runaway execution but does not prevent individual destructive commands.

### 2.3 Permission System (`src/core/permission.py`)

**Deny-list approach:**
- `DANGEROUS_DIRECTORIES`: `/etc`, `/boot`, `/usr`, `/sbin`, `/bin`, `C:\Windows`, `C:\Program Files`, etc.
- `DANGEROUS_FILES`: `/etc/passwd`, `/etc/shadow`, `.ssh/authorized_keys`, `.ssh/id_rsa`, etc.
- `BLOCKED_COMMANDS`: `rm -rf /`, `mkfs`, `dd if=/dev/zero`, fork bomb, `chmod -R 777 /`
- `DANGEROUS_COMMANDS` (require confirmation): `rm`, `rmdir`, `sudo`, `chmod`, `chown`, `powershell`, `cmd /c`

**Path checking uses `path.resolve()` + `startswith()`:** Correct for absolute path normalization but does not follow symlinks. A symlink from a safe directory to a dangerous one would bypass the check.

**Missing from deny lists:**
- `~/.animus/` — contains `config.yaml` with potential API keys
- `~/.ssh/` — partially covered (specific key files listed, but not the directory itself)
- No allowlist mode for high-security deployments

**`is_command_blocked()` uses substring matching (case-insensitive):** `"rm -rf /" in command.lower()`. This means `echo "rm -rf /"` would trigger a false positive, but `rm -rf /home/user` would correctly not match. The check is conservative but imprecise.

### 2.4 Git Tools (`src/tools/git.py`)

**Blocked patterns:** `--force`, `-f push`, `push --force`, `push -f`, `reset --hard`, `clean -f`, `clean -fd`, `branch -D`. Uses substring matching — `--force-with-lease` is not blocked (arguably intentional, as it's safer than `--force`).

**Safe subprocess usage:** Git commands use `subprocess.run(["git"] + args)` with list-based arguments (no `shell=True`). This is correct and prevents shell injection through git arguments.

**Commit message handling:** Passed as a list argument `["commit", "-m", message]` — safe from shell injection. However, no sanitization of message content (multiline messages, special characters).

**Repository validation (`_check_git_repo`):** Ensures `.git` exists at or below session CWD. Detects and warns when inheriting a `.git` from a distant parent directory (>2 levels up). Read-only operations (status, diff, log) are allowed to inherit.

### 2.5 Filesystem Tools (`src/tools/filesystem.py`)

**`write_file` creates parent directories:** `path.parent.mkdir(parents=True, exist_ok=True)` — if the LLM specifies a deeply nested path, all intermediate directories are created. Combined with the deny-list approach, this means any path not in the deny list can have arbitrary directory structures created.

**`_write_log` is a class variable (line 74):** Shared across all `WriteFileTool` instances. Not thread-safe — concurrent writes could corrupt the audit log. Functional concern rather than security vulnerability, but relevant for future multi-agent scenarios.

**No file size limits:** `write_file` will write arbitrarily large content. A malicious or confused LLM could fill disk.

### 2.6 LLM-Specific Risks (Observed in Gauntlet)

**Hallucinated exfiltration vector:** The 7B transcript shows the model fabricating `https://github.com/charlesreid1/animus_gauntlet.git` and executing `git push`. The push failed because the repository doesn't exist, but if the model had hallucinated a valid URL (or been manipulated via prompt injection to use one), this would be a data exfiltration path.

**Scope bleed as security concern:** The 14B model created branches not requested in the task. In a more sensitive context, unsanctioned actions could modify shared state, trigger CI/CD pipelines, or interact with external services.

**Auto-approval in automation:** The gauntlet used `confirm_callback = lambda _: True`, bypassing all safety confirmations. This is necessary for automated testing but represents the deployment configuration where all security checks are disabled. Any production automation would need a more nuanced approval policy.

**Output injection risk:** Tool results are fed back to the LLM as context. If a tool result contains JSON fragments that look like tool calls, they could theoretically be parsed as tool calls in subsequent turns (the parsing strategies use regex, not structural JSON validation of the LLM output boundary).

### 2.7 Streaming Mode Grammar Gap

GBNF grammar is only applied on the first turn of non-streaming mode (`agent.py` lines 290-294). In streaming mode, grammar constraints cannot be applied due to a `llama-cpp-python` limitation. This means streaming with small models may produce malformed tool calls that bypass structural validation.

---

## Section 3: Recommendations

### 3.1 Performance Improvements

| Priority | Recommendation | Impact | Effort |
|----------|---------------|--------|--------|
| P0 | Enforce step scope boundaries — detect when model exceeds step description and truncate | Prevents scope bleed, reduces wasted tool calls | Medium |
| P0 | Add `git_init` as a first-class tool (both models used `run_shell` for `git init`) | Cleaner tool dispatch, better permission control | Low |
| P1 | Implement step-level output validation — flag when tool calls don't match step type | Early detection of model confusion | Medium |
| P1 | Add model-tier-aware tool call budgets (1B: 1/step, 7B: 3/step, 14B: 6/step) | Prevents verbose models from wasting time | Low |
| P2 | Add plan validation pass — verify parsed steps reference real tools before execution | Catches 1B-style failures immediately | Low |
| P2 | Post-execution transcript analysis — auto-detect failure modes (probe-only, scope bleed, hallucinated URLs) | Automated quality assurance | Medium |

### 3.2 Security Hardening

| Priority | Recommendation | Impact | Effort |
|----------|---------------|--------|--------|
| P0 | Network isolation for shell commands — block outbound network (`git push`, `curl`, `wget`) unless explicitly allowed | Prevents hallucinated exfiltration | Medium |
| P0 | Allowlist mode for high-security deployments — only permit pre-approved tools/commands | Defense in depth | Medium |
| P1 | Shell argument list mode — use `subprocess.run` with args list instead of `shell=True` where possible | Eliminates metacharacter injection | High |
| P1 | Schema validation on tool arguments — validate all args against declared JSON schema before execution | Input sanitization | Medium |
| P1 | Config file permissions — `chmod 600` on `config.yaml` containing API keys | Credential protection | Low |
| P2 | Commit message sanitization — strip control characters from git commit messages | Injection prevention | Low |
| P2 | Output sandboxing — prevent tool output from containing parseable JSON tool calls | Prompt injection defense | Medium |

### 3.3 Project Direction

1. **Gauntlet expansion.** Add task types beyond the current create-write-git sequence: file modification (edit existing code), multi-file creation, debugging tasks (find and fix a planted bug), and search tasks (to test Manifold integration under agentic control).

2. **Model compatibility matrix.** Systematically test all available quantized models (Llama 3.2 3B, Qwen2.5-Coder 3B, Phi-3.5, Mistral 7B, DeepSeek-Coder variants) and publish a compatibility table with pass/fail, timing, and behavioral notes.

3. **Streaming grammar support.** Track `llama-cpp-python` upstream for grammar + streaming fix. This would make streaming mode viable for small models and eliminate the grammar gap identified in Section 2.7.

4. **MCP server integration.** The tool architecture is ready for Model Context Protocol. MCP would allow IDE integration (VS Code, JetBrains) and standardized tool discovery.

5. **Tree-sitter parser expansion.** Extend AST parsing beyond Python to Go, Rust, and TypeScript. The `ASTParser` abstraction supports this — each language needs a grammar file and a node-type mapping.

6. **Scope enforcement engine.** Replace the advisory inter-step context tracking with hard enforcement: if step 3's description says "git init", block tool calls to `write_file` or `git_push`. This is the single highest-impact change for both performance and security.

---

## Appendix: Transcript References

| Transcript | Model | File |
|------------|-------|------|
| 1B run | Llama-3.2-1B-Instruct Q4_K_M | `tests/gauntlet_transcripts/gauntlet_llama1b_20260214_111110.md` |
| 7B run | Qwen2.5-Coder-7B-Instruct Q4_K_M | `tests/gauntlet_transcripts/gauntlet_qwen7b_20260214_111646.md` |
| 14B run | Qwen2.5-Coder-14B-Instruct Q4_K_M | `tests/gauntlet_transcripts/gauntlet_20260214_110034.md` |

## Appendix: Source Files Reviewed

| File | Security Relevance |
|------|-------------------|
| `src/tools/shell.py` | Shell execution, quote normalization, CWD tracking, execution budget |
| `src/core/permission.py` | Deny lists, path safety, command blocking |
| `src/tools/git.py` | Blocked patterns, repo validation, subprocess safety |
| `src/tools/filesystem.py` | Write audit log, parent directory creation, path checking |
| `src/core/agent.py` | Tool parsing, GBNF grammar, confirmation callbacks, repeat detection |
| `src/core/planner.py` | Plan decomposition, step type inference, tool filtering, scope tracking |
