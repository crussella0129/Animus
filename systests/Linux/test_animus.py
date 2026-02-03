#!/usr/bin/env python3.11
"""
Animus Functionality Test Script for Linux
Tests: basic response, file creation, coding, code execution
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config import ConfigManager
from src.llm import get_default_provider
from src.core import Agent, AgentConfig

# Test output file
TEST_OUTPUT_FILE = Path(__file__).parent / "test_output.log"

def log(msg: str):
    """Log to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(TEST_OUTPUT_FILE, "a") as f:
        f.write(line + "\n")

async def auto_confirm(tool_name: str, description: str) -> bool:
    """Auto-confirm all tool calls for testing."""
    log(f"  [AUTO-CONFIRM] Tool: {tool_name}")
    return True

async def run_single_query(agent: Agent, prompt: str) -> str:
    """Run a single query and collect the full response."""
    full_response = ""
    try:
        async for turn in agent.run(prompt, stream=False):
            # Turn has 'content' not 'response'
            if turn.role == "assistant" and turn.content:
                full_response += turn.content
    except Exception as e:
        full_response = f"ERROR: {e}"
    return full_response

async def run_tests():
    """Run Animus functionality tests."""
    log("=" * 60)
    log("ANIMUS LINUX FUNCTIONALITY TEST")
    log("=" * 60)

    # Initialize
    log("\n--- Initializing Animus ---")
    try:
        config_mgr = ConfigManager()
        config = config_mgr.config
        provider = get_default_provider(config)

        if not provider.is_available:
            log("ERROR: Provider not available")
            return False

        log(f"Provider: {config.model.provider}")
        log(f"Model available: {provider.is_available}")

        # Create agent config - use the actual GGUF model name
        agent_config = AgentConfig(
            model="qwen2.5-coder-7b-instruct-q4_k_m.gguf",  # Actual model filename
            max_turns=5,
            max_context_tokens=4096,
            require_tool_confirmation=False,  # Auto-confirm for testing
        )

        agent = Agent(
            provider=provider,
            config=agent_config,
            animus_config=config,
            confirm_callback=auto_confirm,
        )

        log("Agent initialized successfully")

    except Exception as e:
        log(f"ERROR initializing: {e}")
        import traceback
        log(traceback.format_exc())
        return False

    test_results = {}

    # Test 1: Basic response
    log("\n--- TEST 1: Basic Response ---")
    try:
        prompt = "What is 2 + 2? Answer with just the number, nothing else."
        log(f"Prompt: {prompt}")
        response = await run_single_query(agent, prompt)
        log(f"Response: {response[:500]}")
        test1_pass = "4" in str(response)
        log(f"TEST 1: {'PASS' if test1_pass else 'FAIL'}")
        test_results["Basic Response"] = test1_pass
    except Exception as e:
        log(f"TEST 1 ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        test_results["Basic Response"] = False

    # Test 2: Code generation
    log("\n--- TEST 2: Code Generation ---")
    try:
        prompt = "Write a Python function called 'is_prime' that checks if a number is prime. Just the function code."
        log(f"Prompt: {prompt[:60]}...")
        response = await run_single_query(agent, prompt)
        log(f"Response: {response[:500]}")
        test2_pass = "def" in str(response) and ("prime" in str(response).lower() or "is_prime" in str(response))
        log(f"TEST 2: {'PASS' if test2_pass else 'FAIL'}")
        test_results["Code Generation"] = test2_pass
    except Exception as e:
        log(f"TEST 2 ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        test_results["Code Generation"] = False

    # Test 3: File reading
    log("\n--- TEST 3: File Reading ---")
    try:
        readme_path = Path(__file__).parent.parent.parent / "README.md"
        prompt = f"Read the first 10 lines of the file at {readme_path} and tell me what this project is."
        log(f"Prompt: {prompt[:60]}...")
        response = await run_single_query(agent, prompt)
        log(f"Response: {response[:500]}")
        # Check if response mentions Animus or coding agent
        test3_pass = "animus" in str(response).lower() or "agent" in str(response).lower() or "cli" in str(response).lower()
        log(f"TEST 3: {'PASS' if test3_pass else 'FAIL'}")
        test_results["File Reading"] = test3_pass
    except Exception as e:
        log(f"TEST 3 ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        test_results["File Reading"] = False

    # Test 4: File creation
    log("\n--- TEST 4: File Creation ---")
    try:
        test_file = Path(__file__).parent / "hello_test.py"
        # Remove if exists from previous run
        if test_file.exists():
            test_file.unlink()

        prompt = f"""Create a Python file at {test_file} with this exact content:
def hello():
    return "Hello from Animus!"

if __name__ == "__main__":
    print(hello())
"""
        log(f"Prompt: Create file at {test_file}")
        response = await run_single_query(agent, prompt)
        log(f"Response: {response[:500]}")

        # Check if file was created
        test4_pass = test_file.exists()
        if test4_pass:
            content = test_file.read_text()
            log(f"File created with content:\n{content[:200]}")
        else:
            log("File was NOT created")
        log(f"TEST 4: {'PASS' if test4_pass else 'FAIL'}")
        test_results["File Creation"] = test4_pass
    except Exception as e:
        log(f"TEST 4 ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        test_results["File Creation"] = False

    # Test 5: Shell command execution
    log("\n--- TEST 5: Shell Command Execution ---")
    try:
        prompt = "Run the command 'python3.11 --version' and tell me the output."
        log(f"Prompt: {prompt}")
        response = await run_single_query(agent, prompt)
        log(f"Response: {response[:500]}")
        test5_pass = "python" in str(response).lower() or "3.11" in str(response)
        log(f"TEST 5: {'PASS' if test5_pass else 'FAIL'}")
        test_results["Shell Command"] = test5_pass
    except Exception as e:
        log(f"TEST 5 ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        test_results["Shell Command"] = False

    # Summary
    log("\n" + "=" * 60)
    log("TEST SUMMARY")
    log("=" * 60)

    for test_name, passed in test_results.items():
        status = "PASS" if passed else "FAIL"
        log(f"  {test_name}: {status}")

    total_pass = sum(test_results.values())
    total_tests = len(test_results)
    log(f"\nTotal: {total_pass}/{total_tests} tests passed")
    log("=" * 60)

    return total_pass >= 3  # Consider success if at least 3 tests pass

if __name__ == "__main__":
    # Clear previous log
    if TEST_OUTPUT_FILE.exists():
        TEST_OUTPUT_FILE.unlink()

    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
