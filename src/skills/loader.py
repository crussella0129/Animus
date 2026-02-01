"""Skill loader for injecting skills into agent context."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.skills.parser import Skill
from src.skills.registry import SkillRegistry


class SkillLoader:
    """Loads and injects skills into agent prompts."""

    def __init__(self, registry: Optional[SkillRegistry] = None):
        """Initialize skill loader.

        Args:
            registry: Skill registry to use. Creates new one if not provided.
        """
        self.registry = registry or SkillRegistry()

    def discover_all(self, project_dir: Optional[Path] = None) -> list[Skill]:
        """Discover all available skills.

        Args:
            project_dir: Project directory for local skills

        Returns:
            List of discovered skills
        """
        self.registry.discover(project_dir)
        return self.registry.list_enabled()

    def load_skill(self, name: str) -> Optional[Skill]:
        """Load a specific skill by name.

        Args:
            name: Skill name

        Returns:
            Skill if found, None otherwise
        """
        return self.registry.get(name)

    def load_skills(self, names: list[str]) -> list[Skill]:
        """Load multiple skills by name.

        Args:
            names: List of skill names

        Returns:
            List of found skills (missing skills are skipped)
        """
        skills = []
        for name in names:
            skill = self.registry.get(name)
            if skill and skill.metadata.enabled:
                skills.append(skill)
        return skills

    def build_prompt_injection(
        self,
        skills: list[Skill],
        include_examples: bool = True,
        include_guidelines: bool = True,
    ) -> str:
        """Build a prompt injection string from skills.

        Args:
            skills: Skills to include
            include_examples: Include examples section
            include_guidelines: Include guidelines section

        Returns:
            Formatted prompt injection string
        """
        if not skills:
            return ""

        parts = ["# Active Skills\n"]

        for skill in skills:
            parts.append(f"\n## {skill.name}\n")
            parts.append(f"\n{skill.description}\n")
            parts.append(f"\n{skill.instructions}\n")

            if include_examples and skill.examples:
                parts.append("\n### Examples\n")
                for example in skill.examples:
                    parts.append(f"- {example}\n")

            if include_guidelines and skill.guidelines:
                parts.append("\n### Guidelines\n")
                for guideline in skill.guidelines:
                    parts.append(f"- {guideline}\n")

        return "".join(parts)

    def inject_into_prompt(
        self,
        base_prompt: str,
        skills: list[Skill],
        position: str = "after",
    ) -> str:
        """Inject skills into an existing prompt.

        Args:
            base_prompt: Original system prompt
            skills: Skills to inject
            position: Where to inject ("before", "after", or "replace")

        Returns:
            Modified prompt with skills injected
        """
        if not skills:
            return base_prompt

        skill_prompt = self.build_prompt_injection(skills)

        if position == "before":
            return f"{skill_prompt}\n\n{base_prompt}"
        elif position == "after":
            return f"{base_prompt}\n\n{skill_prompt}"
        elif position == "replace":
            return skill_prompt
        else:
            raise ValueError(f"Invalid position: {position}")

    def check_requirements(self, skill: Skill, available_tools: list[str]) -> list[str]:
        """Check if skill requirements are met.

        Args:
            skill: Skill to check
            available_tools: List of available tool names

        Returns:
            List of missing requirements
        """
        missing = []
        for req in skill.metadata.requires:
            if req not in available_tools:
                missing.append(req)
        return missing

    def get_compatible_skills(
        self,
        available_tools: list[str],
        project_dir: Optional[Path] = None,
    ) -> list[Skill]:
        """Get skills whose requirements are satisfied.

        Args:
            available_tools: List of available tool names
            project_dir: Project directory for local skills

        Returns:
            List of compatible skills
        """
        all_skills = self.discover_all(project_dir)
        compatible = []

        for skill in all_skills:
            missing = self.check_requirements(skill, available_tools)
            if not missing:
                compatible.append(skill)

        return compatible


def load_skill_for_agent(
    skill_name: str,
    project_dir: Optional[Path] = None,
) -> Optional[str]:
    """Convenience function to load a skill and get its prompt.

    Args:
        skill_name: Name of skill to load
        project_dir: Project directory for local skills

    Returns:
        Skill prompt string, or None if not found
    """
    loader = SkillLoader()
    loader.discover_all(project_dir)

    skill = loader.load_skill(skill_name)
    if skill:
        return skill.to_prompt()
    return None


def get_available_skills(project_dir: Optional[Path] = None) -> list[str]:
    """Get list of available skill names.

    Args:
        project_dir: Project directory for local skills

    Returns:
        List of skill names
    """
    loader = SkillLoader()
    skills = loader.discover_all(project_dir)
    return [s.name for s in skills]
