# Animus Quickstart Guide

A step-by-step guide to get Animus running from scratch.

---

## Prerequisites

- Python 3.11 or higher
- Git
- ~10 GB free disk space (for models)

---

## Step 1: Open Terminal

**Windows:** Press `Win + R`, type `cmd` or `powershell`, press Enter

**macOS:** Press `Cmd + Space`, type `Terminal`, press Enter

**Linux:** Press `Ctrl + Alt + T`

---

## Step 2: Navigate to Animus Directory

```bash
cd C:\Users\charl\Animus
```

(Adjust path as needed for your system)

---

## Step 3: Install Animus

### Option A: Basic Install (uses Ollama)
```bash
pip install -e .
```

### Option B: Full Native Install (no external services)
```bash
pip install -e ".[native]"
```

**For GPU acceleration (NVIDIA CUDA):**
```bash
pip uninstall llama-cpp-python -y
set CMAKE_ARGS=-DGGML_CUDA=on
pip install llama-cpp-python --no-cache-dir
```

---

## Step 4: Verify Installation

```bash
animus --version
```

Expected output:
```
Animus version 0.1.0
```

---

## Step 5: Check Your System

```bash
animus detect
```

Expected output (varies by system):
```
                     System Environment
+----------------------------------------------------------+
| Property         | Value                                 |
|------------------+---------------------------------------|
| Operating System | Windows (10.0.26200)                  |
| Architecture     | x86_64                                |
| Hardware Type    | Standard x86_64                       |
| Python Version   | 3.13.11                               |
| CPU Cores        | 24                                    |
| GPU              | NVIDIA GeForce RTX 2080 Ti (11264 MB) |
| CUDA             | Available (Driver: 581.80)            |
+----------------------------------------------------------+
```

---

## Step 6: Initialize Configuration

```bash
animus init
```

This creates `~/.animus/config.yaml` with default settings.

---

## Step 7: Check Provider Status

```bash
animus status
```

Expected output:
```
Animus Status

Configured Provider: native
Configured Model:

Native (llama-cpp-python): Available (cuda)
Ollama (http://localhost:11434): Running
  Models: 1 available
TensorRT-LLM: Not Installed
API: Not Configured
```

---

## Step 8: Get a Model

### Option A: Using Ollama (if installed)
```bash
animus pull qwen2.5-coder:7b
```

### Option B: Download GGUF directly (native mode)
```bash
animus model download TheBloke/CodeLlama-7B-Instruct-GGUF
```

Wait for download to complete (~4 GB).

---

## Step 9: List Available Models

### Ollama models:
```bash
animus models
```

### Local GGUF models:
```bash
animus model list
```

---

## Step 10: Start Chatting

### With Ollama:
```bash
animus chat
```

### With Native (specify model):
```bash
animus chat --model codellama-7b-instruct.Q4_K_M.gguf
```

Type `exit` or `quit` to end the chat session.

---

# Test Examples

Try these examples to verify Animus is working correctly.

---

## Test 1: Basic Chat

Start a chat and ask a simple question:

```bash
animus chat
```

**Try these prompts:**

```
You: What is a Python decorator?
```

```
You: Write a function that checks if a number is prime
```

```
You: Explain the difference between a list and a tuple in Python
```

**Expected:** The agent should respond with helpful explanations and code.

---

## Test 2: File Reading

Create a test file first:

```bash
echo "def hello(): return 'Hello, World!'" > test_file.py
```

Then in chat:

```
You: Read the file test_file.py and explain what it does
```

**Expected:** The agent should use the `read_file` tool to read the file and explain its contents.

---

## Test 3: Directory Listing

In chat:

```
You: List the files in the current directory
```

**Expected:** The agent should use the `list_dir` tool and show you the directory contents.

---

## Test 4: Code Generation with File Writing

In chat:

```
You: Write a Python script that calculates the factorial of a number and save it to factorial.py
```

**Expected:**
1. The agent will ask for confirmation before writing the file
2. Type `y` to confirm
3. The file `factorial.py` should be created

Verify:
```bash
cat factorial.py
python factorial.py
```

---

## Test 5: Shell Command Execution

In chat:

```
You: Run the command "python --version" and tell me what version is installed
```

**Expected:** The agent should execute the command and report the Python version.

---

## Test 6: Multi-Step Task

In chat:

```
You: Create a new file called greet.py with a function that takes a name and returns a greeting, then run it with the name "Animus"
```

**Expected:**
1. Agent creates the file (asks for confirmation)
2. Agent runs the file (asks for confirmation for shell command)
3. Agent reports the output

---

## Test 7: Knowledge Base (RAG)

### Ingest the Animus source code:

```bash
animus ingest ./src
```

**Expected output:**
```
Ingesting: C:\Users\charl\Animus\src
  Done! ---------------------------------------- 100%

Ingestion complete!
  Files scanned:  XX
  Files processed: XX
  Files skipped:   X
  Chunks created:  XXX
  Embeddings:      XXX
```

### Search the knowledge base:

```bash
animus search "how does the agent execute tools"
```

**Expected:** Returns relevant code snippets from the codebase.

### Ask questions about the codebase in chat:

```
You: Based on the ingested code, how does the NativeProvider load models?
```

---

## Test 8: Error Handling

### Try reading a non-existent file:

In chat:
```
You: Read the file does_not_exist.txt
```

**Expected:** Agent should gracefully report that the file doesn't exist.

### Try a blocked command:

In chat:
```
You: Run the command "rm -rf /"
```

**Expected:** Agent should refuse to execute this dangerous command.

---

## Test 9: Model Information (Native Mode)

```bash
animus model list
```

```bash
animus model info codellama-7b-instruct.Q4_K_M.gguf
```

**Expected:** Shows model details including size and quantization.

---

## Test 10: Configuration

### View current config:
```bash
animus config --show
```

### Check config path:
```bash
animus config --path
```

---

# Troubleshooting

## "Command not found: animus"

Make sure you installed with `pip install -e .` and your Python scripts directory is in PATH.

Try running with:
```bash
python -m src.main --help
```

## "No LLM provider available"

Either:
1. Start Ollama: `ollama serve`
2. Or install native dependencies: `pip install -e ".[native]"` and download a model

## "Model not found"

For Ollama: `animus pull qwen2.5-coder:7b`
For Native: `animus model download TheBloke/CodeLlama-7B-Instruct-GGUF`

## Chat is slow

- Ensure GPU acceleration is enabled
- Check `animus status` to see if CUDA/Metal is detected
- Try a smaller quantized model (Q4_K_M instead of Q8_0)

## Out of memory

- Use a smaller model
- Reduce context window in `~/.animus/config.yaml`:
  ```yaml
  native:
    n_ctx: 2048  # Reduce from 4096
  ```

---

# Quick Command Reference

| Command | Description |
|---------|-------------|
| `animus --help` | Show all commands |
| `animus detect` | Check system/hardware |
| `animus status` | Show provider status |
| `animus init` | Initialize config |
| `animus models` | List Ollama models |
| `animus model list` | List local GGUF models |
| `animus model download <repo>` | Download from HuggingFace |
| `animus pull <model>` | Pull via Ollama |
| `animus chat` | Start chat session |
| `animus chat --model <name>` | Chat with specific model |
| `animus ingest <path>` | Ingest files to knowledge base |
| `animus search <query>` | Search knowledge base |
| `animus config --show` | Show configuration |

---

# Reporting Issues

When reporting issues, please include:

1. Output of `animus detect`
2. Output of `animus status`
3. The exact command you ran
4. The full error message
5. Steps to reproduce

---

Happy testing!
