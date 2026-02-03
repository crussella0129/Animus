# Animus Linux Installation & Functionality Test Log

**Test Date:** 2026-02-02
**Tester:** Claude Code (Opus 4.5)

---

## System Information

### Hardware
| Component | Details |
|-----------|---------|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics |
| Cores/Threads | 8 cores / 16 threads |
| Max Clock | 5.137 GHz |
| Architecture | x86_64 |
| RAM | 14 GB total (~11 GB available) |
| GPU | AMD Radeon 780M (Integrated) |
| L3 Cache | 16 MB |

### Firmware/BIOS
| Property | Value |
|----------|-------|
| Vendor | Lenovo |
| Device | IdeaPad 5 2-in-1 16AHP9 |

### Operating System
| Property | Value |
|----------|-------|
| OS | Ubuntu 22.04.5 LTS (Jammy Jellyfish) |
| Kernel | 6.8.0-94-generic |
| Architecture | x86_64 |

### Initial Software State
| Component | Status |
|-----------|--------|
| Python | 3.10.12 (/usr/bin/python3) |
| pip | NOT INSTALLED |
| git | Installed |

---

## Installation Process

### Step 1: Install pip (NOT IN README - FRESH INSTALL REQUIREMENT)

**Issue:** Fresh Ubuntu install does not have pip installed.

**Command:**
```bash
sudo apt update && sudo apt install -y python3-pip
```

**Output:**
```
Successfully installed pip 22.0.2
pip 22.0.2 from /usr/lib/python3/dist-packages/pip (python 3.10)
```

**Status:** SUCCESS

---

### Step 2: Navigate to Animus Directory

**Command:**
```bash
cd /home/charles/Git_Repos/Animus
```

**Status:** SUCCESS

---

### Step 3: Run Animus Installer (FIRST ATTEMPT - FAILED)

**Command:**
```bash
python3 install.py
```

**Output:**
```
Error: Python 3.11+ required, found 3.10.12 (main, Jan  8 2026, 06:52:19) [GCC 11.4.0]
Please install Python 3.11 or later.
```

**Status:** FAILED - Python version too old

**Issue:** Ubuntu 22.04 LTS ships with Python 3.10, but Animus requires Python 3.11+

---

### Step 4: Install Python 3.11 (NOT IN README - FRESH INSTALL REQUIREMENT)

**Issue:** Need to install Python 3.11+ from deadsnakes PPA

**Commands:**
```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

**Output:**
```
Python 3.11.14
/usr/bin/python3.11
```

**Status:** SUCCESS

---

### Step 5: Run Animus Installer with Python 3.11

**Command:**
```bash
python3.11 install.py --verbose
```

**Output:**
```
Installing Animus
  Hardware: standard_x86
  ✓ Installing base dependencies...
  ✓ Installing native inference backend... (FAILED - missing build tools)
  ✓ Installing embedding model support...
  ✓ Configuring Animus...
  ✓ Verification complete

Installation complete!
Warnings:
  ! Native backend installation failed
```

**Status:** PARTIAL SUCCESS - Base installed, native backend failed

---

### Step 6: Install Build Dependencies (NOT IN README - FRESH INSTALL REQUIREMENT)

**Issue:** llama-cpp-python requires build tools to compile

**Commands:**
```bash
sudo apt install -y build-essential cmake ninja-build
```

**Status:** SUCCESS

---

### Step 7: Install llama-cpp-python Manually

**Command:**
```bash
python3.11 -m pip install llama-cpp-python --no-cache-dir
```

**Output:**
```
Successfully built llama-cpp-python
Successfully installed diskcache-5.6.3 llama-cpp-python-0.3.16
```

**Status:** SUCCESS

---

### Step 8: Verify Installation

**Command:** `animus detect`
```
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Property         ┃ Value                ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ Operating System │ Linux (Ubuntu 22.04) │
│ Architecture     │ x86_64               │
│ Hardware Type    │ Standard x86_64      │
│ Python Version   │ 3.11.14              │
│ CPU Cores        │ 16                   │
│ GPU              │ Not detected         │
└──────────────────┴──────────────────────┘
```

**Command:** `animus status`
```
Animus Status

Configured Provider: native
Configured Model: (auto-detect)

Native (llama-cpp-python): Available (cpu)
  No local models found
TensorRT-LLM: Not Installed
API: Not Configured
```

**Status:** SUCCESS - Native backend available (CPU mode)

**Note:** AMD Radeon 780M integrated GPU not detected. ROCm setup would be required for GPU acceleration but is complex. CPU mode sufficient for testing.

---

## Model Selection

### Hardware Considerations:
- 14 GB RAM (~11 GB available)
- CPU-only inference (no GPU acceleration)
- 16 threads available for parallel processing

### Animus Requirements:
- JSON output capability (for tool calling)
- Coding proficiency
- Instruction following

### Selected Model: `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`

**Rationale:**
1. Qwen2.5-Coder is excellent at structured/JSON output
2. Specifically trained for coding tasks
3. 7B parameter size fits comfortably in 14GB RAM
4. Q4_K_M quantization provides good quality/speed balance
5. Strong instruction-following capability

---

## Model Download

**Command:**
```bash
animus pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
```

**Output:**
```
Downloading model: Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
Target directory: /home/charles/.animus/models

Download complete!
  Path: /home/charles/.animus/models/qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf
  Size: 3808.2 MB
```

**Model Verification:**
```
animus models

Local GGUF Models
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Name                                            ┃ Size    ┃ Quantization ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002 │ 3.72 GB │ Q4_K_M       │
└─────────────────────────────────────────────────┴─────────┴──────────────┘
```

**Status:** SUCCESS

---

### Step 9: Model Download Issue - Downloaded Split File

**Issue:** `animus pull` downloaded a split file (00001-of-00002) instead of a single complete file.

**Problem:**
```
qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf (3.72 GB)
```
This is part 1 of 2 of a split model - incomplete!

**Solution:** Download the single-file version directly using huggingface_hub

**Commands:**
```bash
# Remove incomplete split file
rm ~/.animus/models/qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf

# Download single-file version
python3.11 -c "
from huggingface_hub import hf_hub_download
import os
path = hf_hub_download(
    repo_id='Qwen/Qwen2.5-Coder-7B-Instruct-GGUF',
    filename='qwen2.5-coder-7b-instruct-q4_k_m.gguf',
    local_dir=os.path.expanduser('~/.animus/models'),
    local_dir_use_symlinks=False
)
print(f'Downloaded to: {path}')
"
```

**Result:**
```
qwen2.5-coder-7b-instruct-q4_k_m.gguf (4.36 GB)
```

**Status:** SUCCESS - Single-file model downloaded

---

## Functionality Testing

### Test Environment Setup
```bash
cd /home/charles/Git_Repos/Animus/systests/Linux
python3.11 test_animus.py
```

### Test Results

| Test | Description | Status |
|------|-------------|--------|
| TEST 1 | Basic Response (2+2=?) | PASS |
| TEST 2 | Code Generation (is_prime function) | PASS |
| TEST 3 | File Reading (README.md) | PASS |
| TEST 4 | File Creation (hello_test.py) | PASS |
| TEST 5 | Shell Command Execution (python3.11 --version) | PASS |

**Overall Result: 5/5 tests passed**

### Detailed Test Output

#### TEST 1: Basic Response
```
Prompt: What is 2 + 2? Answer with just the number, nothing else.
Response: 4
```

#### TEST 2: Code Generation
```
Prompt: Write a Python function called 'is_prime' that checks if a number is prime.
Response:
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```

#### TEST 3: File Reading
```
Prompt: Read the first 10 lines of README.md
Response: The Animus project is a local CLI coding agent powered by GGUF models.
It allows you to read and analyze code, write and modify files, execute commands,
and learn from your codebase.
```

#### TEST 4: File Creation
```
Prompt: Create hello_test.py with hello() function
Result: File created successfully with correct content
```

#### TEST 5: Shell Command Execution
```
Prompt: Run 'python3.11 --version'
Result: Command executed successfully
```

---

## Performance Notes

- Model loading: ~20 seconds (CPU, no GPU acceleration)
- Simple response (2+2): ~25 seconds
- Code generation: ~15-30 seconds
- File operations with tool calls: ~30-60 seconds

**Note:** These times are for CPU-only inference on AMD Ryzen 7 8845HS. GPU acceleration with ROCm would significantly improve performance.

---

## Conclusion

### Installation Summary

| Step | Status | Notes |
|------|--------|-------|
| Clone repository | SUCCESS | |
| Install pip | SUCCESS | Not pre-installed on fresh Ubuntu |
| Install Python 3.11 | SUCCESS | Required - Ubuntu 22.04 has 3.10 |
| Run installer | PARTIAL | Base deps OK, native backend failed |
| Install build tools | SUCCESS | build-essential, cmake, ninja-build |
| Install llama-cpp-python | SUCCESS | Manual installation required |
| Download model | WORKAROUND | Split file issue - used direct download |
| Run tests | SUCCESS | 5/5 tests passed |

### New Requirements Identified for README

The following steps were NOT documented but were required for a fresh Ubuntu 22.04 install:

1. **Install pip:** `sudo apt install python3-pip`
2. **Install Python 3.11:** From deadsnakes PPA
3. **Install build tools:** `sudo apt install build-essential cmake ninja-build`
4. **Manual llama-cpp-python install:** If auto-install fails

### Bugs Found

1. **Model download downloads split files instead of single files**
   - Impact: Model fails to load
   - Workaround: Direct download via huggingface_hub

### Model Recommendation

For systems similar to this (14GB RAM, no NVIDIA GPU, CPU inference):
- **Model:** Qwen2.5-Coder-7B-Instruct
- **Quantization:** Q4_K_M (4.36 GB)
- **Rationale:** Excellent JSON output, purpose-built for coding, fits in memory

### Final Status

**ANIMUS INSTALLATION: SUCCESSFUL (with documented workarounds)**
**FUNCTIONALITY TESTS: 5/5 PASSED**

---

## Web Search Security Implementation (Ungabunga-Box)

### Phase 1: Rule-Based Validation (Completed)

Created secure web search capability with process isolation:

**Files Created:**
- `src/tools/web.py` - WebSearchTool and WebFetchTool
- `tests/test_web_tools.py` - 24 passing tests

**Security Features:**
- 30+ regex patterns for prompt injection detection
- Process isolation with `subprocess` and `env={}` (no credential leakage)
- HTML sanitization with bleach
- Suspicious URL detection (javascript:, file:, data:text/html, vbscript:)
- Human confirmation for suspicious content

**Test Results:** 24 passed, 2 skipped (network tests)

### Phase 2: LLM Validator (Completed)

Extended the HybridJudge pattern with a smaller model for semantic validation:

**Files Created:**
- `src/core/web_validator.py` - WebContentJudge, WebContentLLMValidator, WebContentRuleEngine
- `tests/test_web_validator.py` - 25 passing tests

**Architecture (Ungabunga-Box Pattern):**
1. **Rules BLOCK** → Content rejected immediately (critical threats)
2. **Rules WARN** → LLM validates (semantic analysis)
3. **LLM SAFE** → Content approved (false positive handling)
4. **LLM UNCERTAIN/UNSAFE** → Human decides

**Validator Model:**
- Model: `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (~1.1 GB)
- Purpose: Different model from main agent = harder to bypass both
- Download instructions added to README

**Test Results:** 25 passed

### Total Web Security Tests: 49 passed, 2 skipped

---

*Log generated by Claude Code (Opus 4.5) on 2026-02-03*
