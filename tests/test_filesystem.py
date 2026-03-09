"""Tests for filesystem tools."""

from pathlib import Path

import pytest

from src.tools.filesystem import WriteFileTool


class TestWriteFileUnescaping:
    """WriteFileTool must unescape JSON-escaped sequences in content."""

    def test_escaped_quotes_are_unescaped(self, tmp_path):
        tool = WriteFileTool()
        path = tmp_path / "test.py"
        # Model sends JSON-escaped content
        result = tool.execute({"path": str(path), "content": '\"\"\"docstring\"\"\"'})
        assert path.read_text() == '"""docstring"""'

    def test_escaped_newlines_are_unescaped(self, tmp_path):
        tool = WriteFileTool()
        path = tmp_path / "test.py"
        result = tool.execute({"path": str(path), "content": 'line1\\nline2'})
        assert path.read_text() == 'line1\nline2'

    def test_escaped_backslash_preserved(self, tmp_path):
        tool = WriteFileTool()
        path = tmp_path / "test.py"
        result = tool.execute({"path": str(path), "content": 'path = C:\\\\Users\\\\foo'})
        assert path.read_text() == 'path = C:\\Users\\foo'

    def test_normal_content_unchanged(self, tmp_path):
        tool = WriteFileTool()
        path = tmp_path / "test.py"
        result = tool.execute({"path": str(path), "content": 'def hello():\n    pass\n'})
        assert path.read_text() == 'def hello():\n    pass\n'
