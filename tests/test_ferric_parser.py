"""Tests for the FerricParser Python wrapper."""
import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


class TestFerricParserInterface:
    def test_ferric_parser_implements_language_parser(self):
        from src.knowledge.parser import FerricParser, LanguageParser
        assert issubclass(FerricParser, LanguageParser)

    def test_ferric_parser_supported_extensions(self):
        from src.knowledge.parser import FerricParser
        parser = FerricParser()
        exts = parser.supported_extensions()
        assert ".py" in exts
        assert ".rs" in exts

    def test_ferric_parser_falls_back_when_binary_absent(self, tmp_path):
        """When ferric-parse binary is not discoverable, parse_file returns empty FileParseResult."""
        from src.knowledge.parser import FerricParser
        from unittest.mock import patch

        sample = tmp_path / "test.py"
        sample.write_text("def hello(): pass\n")

        with patch("src.ferric.find_ferric_binary", return_value=None):
            parser = FerricParser()  # binary_path=None → auto-discovery returns None
            assert parser.is_available() is False
            result = parser.parse_file(sample)

        assert result.file_path == str(sample)
        assert isinstance(result.nodes, list)
        assert len(result.nodes) == 0

    def test_ferric_parser_parses_python_file(self, tmp_path):
        """When binary is available, parse_file returns populated FileParseResult."""
        from src.knowledge.parser import FerricParser

        fake_output = json.dumps({
            "file_path": "/tmp/test.py",
            "nodes": [
                {"kind": "function", "name": "hello", "qualified_name": "test.hello",
                 "file_path": "/tmp/test.py", "line_start": 1, "line_end": 1,
                 "docstring": "", "args": [], "bases": [], "decorators": []}
            ],
            "edges": []
        })

        sample = tmp_path / "test.py"
        sample.write_text("def hello(): pass\n")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=fake_output, stderr=""
            )
            parser = FerricParser(binary_path="ferric-parse")
            result = parser.parse_file(sample)

        assert len(result.nodes) == 1
        assert result.nodes[0].name == "hello"

    def test_is_available_true_when_binary_provided(self):
        from src.knowledge.parser import FerricParser
        parser = FerricParser(binary_path="ferric-parse")
        # binary_path is set, so is_available() returns True
        assert parser.is_available() is True

    def test_is_available_false_when_binary_absent(self):
        from src.knowledge.parser import FerricParser
        parser = FerricParser(binary_path=None)
        # No binary discovered
        with patch("src.ferric.find_ferric_binary", return_value=None):
            parser2 = FerricParser()
            assert parser2.is_available() is False
