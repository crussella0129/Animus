"""SKILL.md parser for Animus skills system.

Skills follow the Anthropic SKILL.md format:
- YAML frontmatter with name and description
- Markdown body with instructions, examples, guidelines
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class SkillMetadata:
    """Skill metadata from YAML frontmatter."""
    name: str
    description: str
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)  # Required tools or dependencies
    enabled: bool = True


@dataclass
class Skill:
    """A complete skill definition."""
    metadata: SkillMetadata
    instructions: str  # The markdown body
    source_path: Optional[Path] = None  # Where the skill was loaded from

    # Parsed sections from markdown
    examples: list[str] = field(default_factory=list)
    guidelines: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Skill name."""
        return self.metadata.name

    @property
    def description(self) -> str:
        """Skill description."""
        return self.metadata.description

    def to_prompt(self) -> str:
        """Convert skill to prompt injection format."""
        prompt = f"## Skill: {self.name}\n\n"
        prompt += f"{self.description}\n\n"
        prompt += self.instructions
        return prompt


class SkillParser:
    """Parser for SKILL.md files."""

    # Regex to match YAML frontmatter
    FRONTMATTER_PATTERN = re.compile(
        r'^---\s*\n(.*?)\n---\s*\n',
        re.DOTALL
    )

    # Section patterns
    EXAMPLES_PATTERN = re.compile(
        r'##\s*Examples?\s*\n(.*?)(?=\n##|\Z)',
        re.DOTALL | re.IGNORECASE
    )
    GUIDELINES_PATTERN = re.compile(
        r'##\s*Guidelines?\s*\n(.*?)(?=\n##|\Z)',
        re.DOTALL | re.IGNORECASE
    )

    def parse_file(self, path: Path) -> Skill:
        """Parse a SKILL.md file.

        Args:
            path: Path to the SKILL.md file

        Returns:
            Parsed Skill object

        Raises:
            ValueError: If file format is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Skill file not found: {path}")

        content = path.read_text(encoding="utf-8")
        skill = self.parse_string(content)
        skill.source_path = path
        return skill

    def parse_string(self, content: str) -> Skill:
        """Parse SKILL.md content from string.

        Args:
            content: SKILL.md file content

        Returns:
            Parsed Skill object

        Raises:
            ValueError: If format is invalid
        """
        # Extract frontmatter
        match = self.FRONTMATTER_PATTERN.match(content)
        if not match:
            raise ValueError("Invalid SKILL.md format: missing YAML frontmatter")

        frontmatter_yaml = match.group(1)
        body = content[match.end():]

        # Parse YAML frontmatter
        try:
            frontmatter = yaml.safe_load(frontmatter_yaml) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML frontmatter: {e}")

        # Validate required fields
        if "name" not in frontmatter:
            raise ValueError("SKILL.md missing required 'name' field in frontmatter")
        if "description" not in frontmatter:
            raise ValueError("SKILL.md missing required 'description' field in frontmatter")

        # Create metadata
        metadata = SkillMetadata(
            name=frontmatter["name"],
            description=frontmatter["description"],
            version=frontmatter.get("version", "1.0.0"),
            author=frontmatter.get("author"),
            tags=frontmatter.get("tags", []),
            requires=frontmatter.get("requires", []),
            enabled=frontmatter.get("enabled", True),
        )

        # Parse sections from body
        examples = self._extract_list_items(
            self.EXAMPLES_PATTERN.search(body)
        )
        guidelines = self._extract_list_items(
            self.GUIDELINES_PATTERN.search(body)
        )

        return Skill(
            metadata=metadata,
            instructions=body.strip(),
            examples=examples,
            guidelines=guidelines,
        )

    def _extract_list_items(self, match: Optional[re.Match]) -> list[str]:
        """Extract list items from a section match."""
        if not match:
            return []

        section_content = match.group(1)
        items = []

        # Match markdown list items (- or *)
        for line in section_content.split("\n"):
            line = line.strip()
            if line.startswith(("- ", "* ", "• ")):
                items.append(line[2:].strip())
            elif line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
                # Numbered list
                items.append(line[2:].strip())

        return items

    def parse_directory(self, directory: Path) -> list[Skill]:
        """Parse all SKILL.md files in a directory.

        Args:
            directory: Directory to search for skills

        Returns:
            List of parsed skills
        """
        skills = []

        if not directory.exists():
            return skills

        # Look for SKILL.md files in subdirectories (skill-name/SKILL.md)
        for skill_dir in directory.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    try:
                        skill = self.parse_file(skill_file)
                        skills.append(skill)
                    except (ValueError, FileNotFoundError) as e:
                        print(f"Warning: Failed to parse {skill_file}: {e}")

        # Also look for standalone .md files named SKILL.md or *.skill.md
        for md_file in directory.glob("*.skill.md"):
            try:
                skill = self.parse_file(md_file)
                skills.append(skill)
            except (ValueError, FileNotFoundError) as e:
                print(f"Warning: Failed to parse {md_file}: {e}")

        return skills


def create_skill_template(name: str, description: str) -> str:
    """Create a SKILL.md template.

    Args:
        name: Skill name (lowercase, hyphen-separated)
        description: Brief skill description

    Returns:
        SKILL.md template content
    """
    return f'''---
name: {name}
description: {description}
version: 1.0.0
tags: []
requires: []
---

# {name.replace("-", " ").title()}

{description}

## Instructions

1. First, understand the user's request
2. Then, apply the skill's expertise
3. Finally, provide clear output

## Examples

- Example usage 1
- Example usage 2
- Example usage 3

## Guidelines

- Follow best practices
- Be thorough but concise
- Ask for clarification if needed
'''
