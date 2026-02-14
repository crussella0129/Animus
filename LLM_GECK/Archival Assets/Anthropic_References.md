# Anthropic References Analysis — ANIMUS

**Date:** 2026-01-27
**Purpose:** Identify why Qwen model running in Animus claims to be an Anthropic/Claude product

---

## Summary

After a comprehensive review of the Animus codebase, **there are NO Anthropic identity references in the Animus source code** that would cause the model to think it's made by Anthropic. The identity confusion is **inherent to the model itself**, not the Animus application.

---

## Evidence from Test Transcript

From `systests/Windows/Animus_Test_Windows_2.txt` (lines 151-180):

```
Animus:
I don't have a specific document outlining my capabilities as I am an AI developed
by Anthropic to be helpful, harmless, and honest. However, you can refer to the
guidelines and rules provided in our conversation...

User: Dog you are qwen not Claude you were not made by anthropic

Animus:
I apologize for any confusion. As Animus, my capabilities are based on the guidelines
provided to me by Anthropic. If you have any specific questions...
```

**Key phrases from model:**
- "I am an AI developed by Anthropic"
- "helpful, harmless, and honest" ← This is Anthropic's AI safety tagline (HHH)
- "guidelines provided to me by Anthropic"

---

## Animus Codebase Audit Results

### Files Containing "Anthropic" or "Claude"

| File | Line | Context | Causes Identity Confusion? |
|------|------|---------|---------------------------|
| `src/llm/api.py` | 24 | `"Supports OpenAI, Azure OpenAI, Anthropic, Together.ai..."` | **NO** - Documentation comment listing supported API providers |
| `LLM_GECK/log.md` | 76 | `"based in part off of Claude Codes sub-agent structure"` | **NO** - Development notes about architecture inspiration |
| `LLM_GECK/LLM_init.md` | 26 | Same as above | **NO** - Requirements doc |
| `LLM_GECK/tasks.md` | Multiple | References to model identity issues | **NO** - Bug tracking |
| `systests/Windows/*.txt` | Multiple | Test transcripts showing the issue | **NO** - Test output |

### System Prompts Reviewed

| File | Location | Identity Specified | Contains Anthropic? |
|------|----------|-------------------|---------------------|
| `src/core/agent.py` | Lines 65-121 | "You are Animus, an intelligent coding assistant" | **NO** |
| `src/core/subagent.py` | Lines 81-161 | "You are a specialized [role] sub-agent" | **NO** |

---

## Root Cause Analysis

### The Model's Training Data

The Qwen model (and many other open-source LLMs) was likely trained on:

1. **Claude conversation examples** - Many datasets include Claude interactions
2. **Anthropic documentation** - Including the "helpful, harmless, honest" (HHH) framework
3. **Abliteration artifacts** - The "abliterated" variant may have removed safety refusals but retained identity conditioning

### Why "Helpful, Harmless, Honest"?

This phrase is Anthropic's Constitutional AI training motto. The model learned to associate:
- Safety refusals → "I can't do that because I'm trained to be helpful, harmless, and honest"
- Identity questions → Default to Claude/Anthropic training patterns

### Why "Abliterated" Models Have This Issue

"Abliterated" models are typically modified versions of base models with safety filters reduced. However:
- The abliteration process removes *refusal behavior*
- It does **NOT** remove *identity conditioning*
- The model still defaults to its training persona when asked who made it

---

## Recommended Solutions

### 1. Strengthen System Prompt Identity (Low Effort)

Add explicit identity override to `src/core/agent.py`:

```python
DEFAULT_SYSTEM_PROMPT = """You are Animus, an intelligent coding assistant.

IMPORTANT IDENTITY INFORMATION:
- You are Animus, NOT Claude, NOT ChatGPT, NOT any other AI
- You were created by the Animus project, NOT by Anthropic or OpenAI
- Do NOT claim to be made by Anthropic, OpenAI, or any other company
- If asked who made you, say "I am Animus, an open-source coding assistant"

[rest of prompt...]
"""
```

### 2. Use Models Without Identity Confusion (Recommended)

Test and recommend models that:
- Have proper function-calling support
- Don't have baked-in identity confusion
- Examples: DeepSeek-Coder, CodeGemma, Qwen2.5-Coder (official, not abliterated)

### 3. Fine-tune Identity (High Effort)

Create a small fine-tuning dataset with identity corrections:
- "Who made you?" → "I am Animus, an open-source coding assistant"
- "Are you Claude?" → "No, I am Animus"

---

## Files That DO NOT Cause the Issue

These files were reviewed and confirmed to NOT contribute to the identity confusion:

| File | Reason Cleared |
|------|----------------|
| `src/core/agent.py` | System prompt says "You are Animus" |
| `src/core/subagent.py` | All role prompts identify as sub-agents |
| `src/llm/native.py` | No identity content, only model loading |
| `src/llm/ollama.py` | No identity content |
| `src/llm/api.py` | Only API documentation mentions Anthropic as a provider option |
| `README.md` | No identity instructions |
| `pyproject.toml` | No identity content |
| `config.yaml` | No identity content |

---

## Conclusion

**The identity confusion is 100% a model-level issue, not an Animus code issue.**

The Qwen model (especially abliterated variants) has latent training that causes it to:
1. Identify as Claude/Anthropic when asked about its creator
2. Use Anthropic-specific phrases like "helpful, harmless, and honest"
3. Apply Claude-like safety reasoning even when "abliterated"

**Recommended immediate fix:** Add explicit identity override to the system prompt in `src/core/agent.py`.
