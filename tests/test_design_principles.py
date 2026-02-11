"""Tests to enforce critical design principles."""

import ast
import pytest
from pathlib import Path


class TestNoUserInputTimeouts:
    """Enforce: Rule #1 - Never Timeout User Input."""

    def test_no_timeout_in_confirm_callback(self):
        """Ensure confirmation prompts have no timeout."""
        main_file = Path("src/main.py")
        with open(main_file, 'r') as f:
            content = f.read()

        # Check _make_confirm_callback function
        assert "_make_confirm_callback" in content
        confirm_section = content[content.find("def _make_confirm_callback"):content.find("def _make_confirm_callback") + 1000]

        # Should NOT contain timeout parameter
        assert "timeout=" not in confirm_section.lower()
        assert "timeout:" not in confirm_section.lower()

    def test_no_timeout_in_main_input(self):
        """Ensure main REPL input has no timeout."""
        main_file = Path("src/main.py")
        with open(main_file, 'r') as f:
            content = f.read()

        # Find the main input loop
        assert 'console.input("[bold cyan]You>[/] ")' in content

        # Get the context around it
        idx = content.find('console.input("[bold cyan]You>[/] ")')
        context = content[idx:idx+200]

        # Should NOT contain timeout
        assert "timeout=" not in context.lower()
        assert "timeout:" not in context.lower()

    def test_input_timeout_documentation(self):
        """Ensure the no-timeout principle is documented."""
        principles_file = Path("docs/DESIGN_PRINCIPLES.md")

        if not principles_file.exists():
            pytest.skip("Design principles doc not yet created")

        with open(principles_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should document the no-timeout rule
        assert "timeout" in content.lower()
        assert "user input" in content.lower()
        assert "never" in content.lower()

    def test_console_input_calls_no_timeout(self):
        """Check all console.input() calls for timeout parameters."""
        main_file = Path("src/main.py")

        with open(main_file, 'r') as f:
            tree = ast.parse(f.read())

        # Find all console.input() calls
        input_calls = []

        class InputCallVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if (isinstance(node.func, ast.Attribute) and
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id == "console" and
                    node.func.attr == "input"):

                    # Check for timeout keyword argument
                    for keyword in node.keywords:
                        if keyword.arg == "timeout":
                            input_calls.append({
                                "line": node.lineno,
                                "has_timeout": True
                            })
                            return

                    input_calls.append({
                        "line": node.lineno,
                        "has_timeout": False
                    })

                self.generic_visit(node)

        visitor = InputCallVisitor()
        visitor.visit(tree)

        # Assert no input calls have timeout
        for call in input_calls:
            assert not call["has_timeout"], f"console.input() at line {call['line']} has timeout parameter (VIOLATION)"


class TestSecurityHardcoded:
    """Enforce: Rule #2 - LLMs Never Make Security Decisions."""

    def test_permission_checker_is_hardcoded(self):
        """Ensure PermissionChecker uses hardcoded rules."""
        from src.core.permission import (
            DANGEROUS_DIRECTORIES,
            DANGEROUS_FILES,
            BLOCKED_COMMANDS,
            DANGEROUS_COMMANDS,
        )

        # These should be frozensets (immutable)
        assert isinstance(DANGEROUS_DIRECTORIES, frozenset)
        assert isinstance(DANGEROUS_FILES, frozenset)
        assert isinstance(BLOCKED_COMMANDS, frozenset)
        assert isinstance(DANGEROUS_COMMANDS, frozenset)

        # Should have reasonable number of rules
        assert len(DANGEROUS_DIRECTORIES) > 0
        assert len(BLOCKED_COMMANDS) > 0
        assert len(DANGEROUS_COMMANDS) > 0

    def test_permission_checker_no_llm_calls(self):
        """Ensure PermissionChecker doesn't call LLMs."""
        permission_file = Path("src/core/permission.py")

        with open(permission_file, 'r') as f:
            content = f.read()

        # Should NOT import LLM providers
        assert "from src.llm" not in content
        assert "import openai" not in content
        assert "import anthropic" not in content

        # Should NOT have generate/complete calls
        assert ".generate(" not in content
        assert ".complete(" not in content
        assert ".create(" not in content


class TestIsolationDefaults:
    """Enforce: Safe isolation defaults."""

    def test_ornstein_safe_defaults(self):
        """Ensure OrnsteinConfig has safe defaults."""
        from src.isolation import OrnsteinConfig

        config = OrnsteinConfig()

        # Should default to restrictive settings
        assert config.allow_write is False  # Read-only by default
        assert config.timeout_seconds > 0   # Always have timeout
        assert config.timeout_seconds <= 60  # Reasonable limit
        assert len(config.blocked_ips) > 0  # Block private IPs by default

    def test_isolation_level_safe_default(self):
        """Ensure default isolation level is safe."""
        from src.isolation import IsolationLevel, execute_with_isolation

        # Default should be NONE (explicit opt-in for overhead)
        # This is tested by the function signature
        import inspect
        sig = inspect.signature(execute_with_isolation)
        default_level = sig.parameters['level'].default

        assert default_level == IsolationLevel.NONE
