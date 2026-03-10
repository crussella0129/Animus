"""Integration tests: verify ferric-parse binary produces correct output via FerricParser."""
import json
import subprocess
from pathlib import Path
import pytest

from src.ferric import find_ferric_binary

_DEBUG_DIR = Path(__file__).parent.parent / "target" / "debug"


def _locate_binary() -> str | None:
    """Find ferric-parse: bundled/PATH first, then cargo debug output."""
    found = find_ferric_binary("ferric-parse")
    if found:
        return found
    for candidate in (_DEBUG_DIR / "ferric-parse.exe", _DEBUG_DIR / "ferric-parse"):
        if candidate.exists():
            return str(candidate)
    return None


_FERRIC_PARSE_BINARY = _locate_binary() or ""
_BINARY_EXISTS = bool(_FERRIC_PARSE_BINARY)


@pytest.mark.skipif(not _BINARY_EXISTS, reason="ferric-parse binary not built (run: cargo build -p ferric-parse)")
class TestFerricParseIntegration:
    def test_parse_real_python_file(self):
        """Parse an actual Animus Python file and verify node count."""
        target = Path(__file__).parent.parent / "src" / "tools" / "base.py"
        result = subprocess.run(
            [_FERRIC_PARSE_BINARY, str(target)],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert "nodes" in data
        classes = [n for n in data["nodes"] if n["kind"] == "class"]
        assert len(classes) >= 3, f"Expected ≥3 classes in base.py, got: {[c['name'] for c in classes]}"

    def test_ferric_parser_wrapper_roundtrip(self, tmp_path):
        """FerricParser wrapper produces correct structure from a real binary call."""
        from src.knowledge.parser import FerricParser
        parser = FerricParser(binary_path=_FERRIC_PARSE_BINARY)
        assert parser.is_available()

        sample = tmp_path / "sample.py"
        sample.write_text(
            "class Foo:\n"
            "    def bar(self): pass\n"
            "\n"
            "def standalone(): pass\n"
        )
        result = parser.parse_file(sample)

        assert result.file_path == str(sample)
        names = [n.name for n in result.nodes]
        assert "Foo" in names, f"Expected 'Foo' in {names}"
        assert "standalone" in names, f"Expected 'standalone' in {names}"

        # Methods inside classes should be kind="method"
        foo_class = next(n for n in result.nodes if n.name == "Foo")
        assert foo_class.kind == "class"
        bar_method = next((n for n in result.nodes if n.name == "bar"), None)
        assert bar_method is not None, f"Expected 'bar' in {names}"
        assert bar_method.kind == "method", f"Expected bar to be 'method', got '{bar_method.kind}'"

    def test_unsupported_extension_returns_empty(self, tmp_path):
        """ferric-parse emits exit 0 + empty nodes for unsupported extensions."""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[package]\nname = \"test\"\n")
        result = subprocess.run(
            [_FERRIC_PARSE_BINARY, str(toml_file)],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["nodes"] == []
