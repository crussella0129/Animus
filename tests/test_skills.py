"""Tests for Skills system (SKILL.md parsing, registry, loading).

Tests cover:
- SKILL.md parsing with YAML frontmatter
- Skill registry discovery and priority
- Skill loader and prompt injection
- Template creation and URL installation
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from src.skills.parser import SkillParser, Skill, SkillMetadata, create_skill_template
from src.skills.registry import SkillRegistry
from src.skills.loader import SkillLoader, load_skill_for_agent, get_available_skills


# =============================================================================
# Parser Tests
# =============================================================================


class TestSkillMetadata:
    """Tests for SkillMetadata dataclass."""

    def test_default_values(self):
        """Test default metadata values."""
        meta = SkillMetadata(name="test", description="A test skill")
        assert meta.name == "test"
        assert meta.description == "A test skill"
        assert meta.version == "1.0.0"
        assert meta.author is None
        assert meta.tags == []
        assert meta.requires == []
        assert meta.enabled is True

    def test_custom_values(self):
        """Test custom metadata values."""
        meta = SkillMetadata(
            name="custom",
            description="Custom skill",
            version="2.0.0",
            author="Test Author",
            tags=["coding", "python"],
            requires=["read_file", "write_file"],
            enabled=False,
        )
        assert meta.version == "2.0.0"
        assert meta.author == "Test Author"
        assert "python" in meta.tags
        assert "write_file" in meta.requires
        assert meta.enabled is False


class TestSkill:
    """Tests for Skill dataclass."""

    def test_skill_properties(self):
        """Test skill property accessors."""
        meta = SkillMetadata(name="my-skill", description="My skill description")
        skill = Skill(metadata=meta, instructions="Do the thing.")

        assert skill.name == "my-skill"
        assert skill.description == "My skill description"

    def test_to_prompt(self):
        """Test skill to_prompt conversion."""
        meta = SkillMetadata(name="code-review", description="Review code for issues")
        skill = Skill(metadata=meta, instructions="1. Read the code\n2. Find bugs")

        prompt = skill.to_prompt()
        assert "## Skill: code-review" in prompt
        assert "Review code for issues" in prompt
        assert "1. Read the code" in prompt


class TestSkillParser:
    """Tests for SkillParser."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return SkillParser()

    @pytest.fixture
    def valid_skill_content(self):
        """Valid SKILL.md content."""
        return '''---
name: test-skill
description: A test skill for unit testing
version: 1.0.0
author: Test Author
tags:
  - testing
  - python
requires:
  - read_file
---

# Test Skill

This is the main instruction body.

## Examples

- Example 1: Do this thing
- Example 2: Do that thing

## Guidelines

- Be thorough
- Be concise
'''

    def test_parse_valid_skill(self, parser, valid_skill_content):
        """Test parsing valid SKILL.md content."""
        skill = parser.parse_string(valid_skill_content)

        assert skill.name == "test-skill"
        assert skill.description == "A test skill for unit testing"
        assert skill.metadata.version == "1.0.0"
        assert skill.metadata.author == "Test Author"
        assert "testing" in skill.metadata.tags
        assert "read_file" in skill.metadata.requires
        assert "main instruction body" in skill.instructions

    def test_parse_extracts_examples(self, parser, valid_skill_content):
        """Test that examples are extracted."""
        skill = parser.parse_string(valid_skill_content)
        assert len(skill.examples) == 2
        assert "Do this thing" in skill.examples[0]

    def test_parse_extracts_guidelines(self, parser, valid_skill_content):
        """Test that guidelines are extracted."""
        skill = parser.parse_string(valid_skill_content)
        assert len(skill.guidelines) == 2
        assert "thorough" in skill.guidelines[0]

    def test_parse_missing_frontmatter(self, parser):
        """Test parsing without frontmatter raises."""
        content = "# Just Markdown\n\nNo frontmatter here."
        with pytest.raises(ValueError, match="missing YAML frontmatter"):
            parser.parse_string(content)

    def test_parse_missing_name(self, parser):
        """Test parsing without name field raises."""
        content = '''---
description: Has description but no name
---

Content here.
'''
        with pytest.raises(ValueError, match="missing required 'name' field"):
            parser.parse_string(content)

    def test_parse_missing_description(self, parser):
        """Test parsing without description field raises."""
        content = '''---
name: no-description
---

Content here.
'''
        with pytest.raises(ValueError, match="missing required 'description' field"):
            parser.parse_string(content)

    def test_parse_invalid_yaml(self, parser):
        """Test parsing with invalid YAML raises."""
        content = '''---
name: test
description: [unclosed bracket
---

Content.
'''
        with pytest.raises(ValueError, match="Invalid YAML"):
            parser.parse_string(content)

    def test_parse_minimal_skill(self, parser):
        """Test parsing minimal valid skill."""
        content = '''---
name: minimal
description: Minimal skill
---

Instructions.
'''
        skill = parser.parse_string(content)
        assert skill.name == "minimal"
        assert skill.metadata.enabled is True  # Default

    def test_parse_disabled_skill(self, parser):
        """Test parsing disabled skill."""
        content = '''---
name: disabled-skill
description: This skill is disabled
enabled: false
---

Instructions.
'''
        skill = parser.parse_string(content)
        assert skill.metadata.enabled is False

    def test_parse_file(self, parser, valid_skill_content, tmp_path):
        """Test parsing from file."""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(valid_skill_content, encoding="utf-8")

        skill = parser.parse_file(skill_file)
        assert skill.name == "test-skill"
        assert skill.source_path == skill_file

    def test_parse_file_not_found(self, parser):
        """Test parsing non-existent file raises."""
        with pytest.raises(FileNotFoundError):
            parser.parse_file(Path("/nonexistent/SKILL.md"))

    def test_parse_directory(self, parser, valid_skill_content, tmp_path):
        """Test parsing all skills in a directory."""
        # Create skill subdirectories
        (tmp_path / "skill-a").mkdir()
        (tmp_path / "skill-a" / "SKILL.md").write_text(valid_skill_content)

        (tmp_path / "skill-b").mkdir()
        (tmp_path / "skill-b" / "SKILL.md").write_text('''---
name: skill-b
description: Second skill
---

Instructions.
''')

        skills = parser.parse_directory(tmp_path)
        assert len(skills) == 2
        names = [s.name for s in skills]
        assert "test-skill" in names
        assert "skill-b" in names

    def test_parse_directory_nonexistent(self, parser):
        """Test parsing nonexistent directory returns empty."""
        skills = parser.parse_directory(Path("/nonexistent/dir"))
        assert skills == []

    def test_parse_numbered_list(self, parser):
        """Test parsing numbered list items."""
        content = '''---
name: numbered
description: Has numbered examples
---

## Examples

1. First example
2. Second example
3. Third example
'''
        skill = parser.parse_string(content)
        # Note: The parser extracts text after the number prefix
        assert len(skill.examples) >= 2


class TestCreateSkillTemplate:
    """Tests for create_skill_template function."""

    def test_basic_template(self):
        """Test creating basic template."""
        template = create_skill_template("my-skill", "My skill description")
        assert "name: my-skill" in template
        assert "description: My skill description" in template
        assert "version: 1.0.0" in template
        assert "## Instructions" in template
        assert "## Examples" in template
        assert "## Guidelines" in template

    def test_template_title_formatting(self):
        """Test template title is properly formatted."""
        template = create_skill_template("code-review", "Review code")
        assert "# Code Review" in template


# =============================================================================
# Registry Tests
# =============================================================================


class TestSkillRegistry:
    """Tests for SkillRegistry."""

    @pytest.fixture
    def skill_content(self):
        """Standard skill content for testing."""
        return '''---
name: test-skill
description: Test skill
---

Instructions.
'''

    @pytest.fixture
    def registry_with_skills(self, tmp_path, skill_content):
        """Create registry with test skills."""
        # Create bundled skill
        bundled = tmp_path / "bundled"
        bundled.mkdir()
        (bundled / "builtin").mkdir()
        (bundled / "builtin" / "SKILL.md").write_text('''---
name: builtin-skill
description: A bundled skill
---

Bundled instructions.
''')

        # Create user skill
        user = tmp_path / "user"
        user.mkdir()
        (user / "user-skill").mkdir()
        (user / "user-skill" / "SKILL.md").write_text('''---
name: user-skill
description: A user skill
---

User instructions.
''')

        return SkillRegistry(
            user_skills_dir=user,
            bundled_skills_dir=bundled,
        )

    def test_discover_bundled_skills(self, registry_with_skills):
        """Test discovering bundled skills."""
        skills = registry_with_skills.discover()
        assert "builtin-skill" in skills

    def test_discover_user_skills(self, registry_with_skills):
        """Test discovering user skills."""
        skills = registry_with_skills.discover()
        assert "user-skill" in skills

    def test_user_overrides_bundled(self, tmp_path):
        """Test user skill overrides bundled with same name."""
        bundled = tmp_path / "bundled"
        bundled.mkdir()
        (bundled / "same-skill").mkdir()
        (bundled / "same-skill" / "SKILL.md").write_text('''---
name: same-skill
description: Bundled version
---

Bundled.
''')

        user = tmp_path / "user"
        user.mkdir()
        (user / "same-skill").mkdir()
        (user / "same-skill" / "SKILL.md").write_text('''---
name: same-skill
description: User version
---

User.
''')

        registry = SkillRegistry(user_skills_dir=user, bundled_skills_dir=bundled)
        skills = registry.discover()

        # User version should win
        assert skills["same-skill"].description == "User version"

    def test_get_skill(self, registry_with_skills):
        """Test getting skill by name."""
        registry_with_skills.discover()
        skill = registry_with_skills.get("builtin-skill")
        assert skill is not None
        assert skill.name == "builtin-skill"

    def test_get_nonexistent_skill(self, registry_with_skills):
        """Test getting nonexistent skill returns None."""
        registry_with_skills.discover()
        skill = registry_with_skills.get("nonexistent")
        assert skill is None

    def test_list_skills(self, registry_with_skills):
        """Test listing all skills."""
        registry_with_skills.discover()
        skills = registry_with_skills.list()
        assert len(skills) >= 2

    def test_list_enabled_skills(self, tmp_path):
        """Test listing only enabled skills."""
        user = tmp_path / "user"
        user.mkdir()

        (user / "enabled").mkdir()
        (user / "enabled" / "SKILL.md").write_text('''---
name: enabled-skill
description: Enabled
enabled: true
---

Content.
''')

        (user / "disabled").mkdir()
        (user / "disabled" / "SKILL.md").write_text('''---
name: disabled-skill
description: Disabled
enabled: false
---

Content.
''')

        registry = SkillRegistry(user_skills_dir=user, bundled_skills_dir=tmp_path / "empty")
        registry.discover()

        enabled = registry.list_enabled()
        names = [s.name for s in enabled]
        assert "enabled-skill" in names
        assert "disabled-skill" not in names

    def test_register_manual(self, registry_with_skills):
        """Test manually registering a skill."""
        meta = SkillMetadata(name="manual", description="Manual skill")
        skill = Skill(metadata=meta, instructions="Manual instructions")

        registry_with_skills.register(skill)
        assert registry_with_skills.get("manual") is not None

    def test_unregister(self, registry_with_skills):
        """Test unregistering a skill."""
        registry_with_skills.discover()
        assert registry_with_skills.unregister("builtin-skill") is True
        assert registry_with_skills.get("builtin-skill") is None
        assert registry_with_skills.unregister("nonexistent") is False

    def test_search_by_name(self, registry_with_skills):
        """Test searching skills by name."""
        registry_with_skills.discover()
        results = registry_with_skills.search("builtin")
        assert len(results) >= 1
        assert any(s.name == "builtin-skill" for s in results)

    def test_search_by_description(self, registry_with_skills):
        """Test searching skills by description."""
        registry_with_skills.discover()
        results = registry_with_skills.search("bundled")
        assert len(results) >= 1

    def test_search_by_tag(self, tmp_path):
        """Test searching skills by tag."""
        user = tmp_path / "user"
        user.mkdir()
        (user / "tagged").mkdir()
        (user / "tagged" / "SKILL.md").write_text('''---
name: tagged-skill
description: Has tags
tags:
  - python
  - coding
---

Content.
''')

        registry = SkillRegistry(user_skills_dir=user, bundled_skills_dir=tmp_path / "empty")
        registry.discover()

        results = registry.search("python")
        assert len(results) == 1
        assert results[0].name == "tagged-skill"

    def test_get_by_tag(self, tmp_path):
        """Test getting skills by tag."""
        user = tmp_path / "user"
        user.mkdir()

        (user / "skill1").mkdir()
        (user / "skill1" / "SKILL.md").write_text('''---
name: skill1
description: First
tags: [tag-a]
---

Content.
''')

        (user / "skill2").mkdir()
        (user / "skill2" / "SKILL.md").write_text('''---
name: skill2
description: Second
tags: [tag-a, tag-b]
---

Content.
''')

        registry = SkillRegistry(user_skills_dir=user, bundled_skills_dir=tmp_path / "empty")
        registry.discover()

        tag_a = registry.get_by_tag("tag-a")
        assert len(tag_a) == 2

        tag_b = registry.get_by_tag("tag-b")
        assert len(tag_b) == 1
        assert tag_b[0].name == "skill2"

    def test_create_skill(self, tmp_path):
        """Test creating a new skill from template."""
        user = tmp_path / "user"
        registry = SkillRegistry(user_skills_dir=user, bundled_skills_dir=tmp_path / "empty")

        skill_path = registry.create("new-skill", "A new skill")
        assert skill_path.exists()
        assert skill_path.name == "SKILL.md"

        # Verify it was registered
        assert registry.get("new-skill") is not None

    def test_create_normalizes_name(self, tmp_path):
        """Test that create normalizes skill name."""
        user = tmp_path / "user"
        registry = SkillRegistry(user_skills_dir=user, bundled_skills_dir=tmp_path / "empty")

        skill_path = registry.create("My Cool Skill", "Description")
        assert "my-cool-skill" in str(skill_path)


# =============================================================================
# Loader Tests
# =============================================================================


class TestSkillLoader:
    """Tests for SkillLoader."""

    @pytest.fixture
    def loader_with_skills(self, tmp_path):
        """Create loader with test skills."""
        user = tmp_path / "user"
        user.mkdir()

        (user / "skill-a").mkdir()
        (user / "skill-a" / "SKILL.md").write_text('''---
name: skill-a
description: Skill A
requires: []
---

Skill A instructions.
''')

        (user / "skill-b").mkdir()
        (user / "skill-b" / "SKILL.md").write_text('''---
name: skill-b
description: Skill B
requires:
  - required_tool
---

Skill B instructions.
''')

        registry = SkillRegistry(user_skills_dir=user, bundled_skills_dir=tmp_path / "empty")
        return SkillLoader(registry)

    def test_discover_all(self, loader_with_skills):
        """Test discovering all skills."""
        skills = loader_with_skills.discover_all()
        assert len(skills) >= 2

    def test_load_skill(self, loader_with_skills):
        """Test loading a specific skill."""
        loader_with_skills.discover_all()
        skill = loader_with_skills.load_skill("skill-a")
        assert skill is not None
        assert skill.name == "skill-a"

    def test_load_skills(self, loader_with_skills):
        """Test loading multiple skills."""
        loader_with_skills.discover_all()
        skills = loader_with_skills.load_skills(["skill-a", "skill-b"])
        assert len(skills) == 2

    def test_load_skills_missing(self, loader_with_skills):
        """Test loading skills with some missing."""
        loader_with_skills.discover_all()
        skills = loader_with_skills.load_skills(["skill-a", "nonexistent"])
        assert len(skills) == 1  # Only skill-a found

    def test_build_prompt_injection(self, loader_with_skills):
        """Test building prompt injection string."""
        skills = loader_with_skills.discover_all()
        prompt = loader_with_skills.build_prompt_injection(skills)

        assert "# Active Skills" in prompt
        assert "skill-a" in prompt
        assert "skill-b" in prompt

    def test_build_prompt_injection_empty(self, loader_with_skills):
        """Test building prompt with no skills returns empty."""
        prompt = loader_with_skills.build_prompt_injection([])
        assert prompt == ""

    def test_inject_into_prompt_after(self, loader_with_skills):
        """Test injecting skills after base prompt."""
        loader_with_skills.discover_all()
        skill = loader_with_skills.load_skill("skill-a")

        base = "You are an AI assistant."
        result = loader_with_skills.inject_into_prompt(base, [skill], position="after")

        assert result.startswith("You are an AI assistant.")
        assert "skill-a" in result

    def test_inject_into_prompt_before(self, loader_with_skills):
        """Test injecting skills before base prompt."""
        loader_with_skills.discover_all()
        skill = loader_with_skills.load_skill("skill-a")

        base = "You are an AI assistant."
        result = loader_with_skills.inject_into_prompt(base, [skill], position="before")

        assert result.endswith("You are an AI assistant.")
        assert "skill-a" in result

    def test_inject_into_prompt_replace(self, loader_with_skills):
        """Test replacing base prompt with skills."""
        loader_with_skills.discover_all()
        skill = loader_with_skills.load_skill("skill-a")

        base = "You are an AI assistant."
        result = loader_with_skills.inject_into_prompt(base, [skill], position="replace")

        assert "You are an AI assistant." not in result
        assert "skill-a" in result

    def test_inject_into_prompt_invalid_position(self, loader_with_skills):
        """Test invalid position raises."""
        loader_with_skills.discover_all()
        skill = loader_with_skills.load_skill("skill-a")

        with pytest.raises(ValueError, match="Invalid position"):
            loader_with_skills.inject_into_prompt("base", [skill], position="invalid")

    def test_check_requirements_satisfied(self, loader_with_skills):
        """Test checking satisfied requirements."""
        loader_with_skills.discover_all()
        skill = loader_with_skills.load_skill("skill-a")

        missing = loader_with_skills.check_requirements(skill, ["any_tool"])
        assert missing == []

    def test_check_requirements_missing(self, loader_with_skills):
        """Test checking missing requirements."""
        loader_with_skills.discover_all()
        skill = loader_with_skills.load_skill("skill-b")  # Requires required_tool

        missing = loader_with_skills.check_requirements(skill, ["other_tool"])
        assert "required_tool" in missing

    def test_get_compatible_skills(self, loader_with_skills):
        """Test getting compatible skills."""
        compatible = loader_with_skills.get_compatible_skills(["other_tool"])

        names = [s.name for s in compatible]
        assert "skill-a" in names  # No requirements
        assert "skill-b" not in names  # Missing required_tool


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_load_skill_for_agent(self, tmp_path):
        """Test load_skill_for_agent function."""
        bundled = tmp_path / "bundled"
        # The function will try default bundled path, so we use actual bundled skills
        # This test verifies the function runs without error
        result = load_skill_for_agent("nonexistent-skill", tmp_path)
        assert result is None  # Skill not found

    def test_get_available_skills(self, tmp_path):
        """Test get_available_skills function."""
        # Uses actual bundled skills
        skills = get_available_skills(tmp_path)
        # Should find the bundled skills
        assert isinstance(skills, list)


# =============================================================================
# Bundled Skills Tests
# =============================================================================


class TestBundledSkills:
    """Tests for bundled skills."""

    @pytest.fixture
    def bundled_registry(self):
        """Create registry pointing to actual bundled skills."""
        return SkillRegistry()

    def test_bundled_skills_exist(self, bundled_registry):
        """Test that bundled skills are discovered."""
        skills = bundled_registry.discover()

        # These should be bundled
        expected = ["code-review", "test-gen", "refactor", "explain", "commit"]
        for name in expected:
            assert name in skills, f"Missing bundled skill: {name}"

    def test_bundled_skills_valid(self, bundled_registry):
        """Test that all bundled skills have valid metadata."""
        skills = bundled_registry.discover()

        for name, skill in skills.items():
            assert skill.name, f"Skill {name} missing name"
            assert skill.description, f"Skill {name} missing description"
            assert skill.instructions, f"Skill {name} missing instructions"

    def test_code_review_skill(self, bundled_registry):
        """Test code-review skill content."""
        bundled_registry.discover()
        skill = bundled_registry.get("code-review")

        assert skill is not None
        assert "review" in skill.description.lower() or "code" in skill.description.lower()

    def test_commit_skill(self, bundled_registry):
        """Test commit skill content."""
        bundled_registry.discover()
        skill = bundled_registry.get("commit")

        assert skill is not None
        prompt = skill.to_prompt()
        assert "commit" in prompt.lower() or "message" in prompt.lower()


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_unicode_in_skill(self, tmp_path):
        """Test skill with Unicode content."""
        user = tmp_path / "user"
        user.mkdir()
        (user / "unicode").mkdir()
        (user / "unicode" / "SKILL.md").write_text('''---
name: unicode-skill
description: Skill with Unicode characters
---

Instructions with emojis and unicode:
Japanese: 日本語テスト
Emoji: Some text here
Symbols: <> and more
''', encoding="utf-8")

        registry = SkillRegistry(user_skills_dir=user, bundled_skills_dir=tmp_path / "empty")
        registry.discover()

        skill = registry.get("unicode-skill")
        assert skill is not None
        assert "Japanese" in skill.instructions

    def test_empty_frontmatter_fields(self):
        """Test skill with empty optional fields."""
        parser = SkillParser()
        content = '''---
name: empty-fields
description: Has empty fields
tags: []
requires: []
---

Content.
'''
        skill = parser.parse_string(content)
        assert skill.metadata.tags == []
        assert skill.metadata.requires == []

    def test_multiline_description(self):
        """Test skill with multiline description."""
        parser = SkillParser()
        content = '''---
name: multiline
description: >
  This is a very long description
  that spans multiple lines
  in YAML format.
---

Content.
'''
        skill = parser.parse_string(content)
        assert "very long description" in skill.description

    def test_deeply_nested_directory(self, tmp_path):
        """Test skill in deeply nested directory."""
        nested = tmp_path / "a" / "b" / "c" / "skill"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text('''---
name: nested-skill
description: Deeply nested
---

Content.
''')

        parser = SkillParser()
        skill = parser.parse_file(nested / "SKILL.md")
        assert skill.name == "nested-skill"

    def test_skill_with_code_blocks(self):
        """Test skill with code blocks in instructions."""
        parser = SkillParser()
        content = '''---
name: code-skill
description: Has code examples
---

## Instructions

Here's some code:

```python
def hello():
    print("Hello, World!")
```

And some more text.
'''
        skill = parser.parse_string(content)
        assert "```python" in skill.instructions
        assert "def hello" in skill.instructions
