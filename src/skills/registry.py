"""Skill registry for discovering and managing skills.

Priority order (higher wins on conflict):
1. Project skills (./skills/ or ./.animus/skills/)
2. User skills (~/.animus/skills/)
3. Bundled skills (shipped with Animus)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.skills.parser import Skill, SkillParser


class SkillRegistry:
    """Registry for managing and discovering skills."""

    def __init__(
        self,
        user_skills_dir: Optional[Path] = None,
        project_skills_dir: Optional[Path] = None,
        bundled_skills_dir: Optional[Path] = None,
    ):
        """Initialize skill registry.

        Args:
            user_skills_dir: User-level skills directory (~/.animus/skills/)
            project_skills_dir: Project-level skills directory (./skills/)
            bundled_skills_dir: Bundled skills directory (shipped with Animus)
        """
        self.user_skills_dir = user_skills_dir or (Path.home() / ".animus" / "skills")
        self.project_skills_dir = project_skills_dir
        self.bundled_skills_dir = bundled_skills_dir or (
            Path(__file__).parent / "bundled"
        )

        self._skills: dict[str, Skill] = {}
        self._parser = SkillParser()

    def discover(self, project_dir: Optional[Path] = None) -> dict[str, Skill]:
        """Discover all available skills.

        Searches in priority order (later sources override earlier):
        1. Bundled skills
        2. User skills (~/.animus/skills/)
        3. Project skills (./skills/ or ./.animus/skills/)

        Args:
            project_dir: Project directory to search for local skills

        Returns:
            Dictionary of skill name -> Skill
        """
        self._skills = {}

        # 1. Load bundled skills (lowest priority)
        if self.bundled_skills_dir and self.bundled_skills_dir.exists():
            for skill in self._parser.parse_directory(self.bundled_skills_dir):
                self._skills[skill.name] = skill

        # 2. Load user skills
        if self.user_skills_dir.exists():
            for skill in self._parser.parse_directory(self.user_skills_dir):
                self._skills[skill.name] = skill

        # 3. Load project skills (highest priority)
        if project_dir:
            # Check ./skills/
            project_skills = project_dir / "skills"
            if project_skills.exists():
                for skill in self._parser.parse_directory(project_skills):
                    self._skills[skill.name] = skill

            # Check ./.animus/skills/
            project_animus_skills = project_dir / ".animus" / "skills"
            if project_animus_skills.exists():
                for skill in self._parser.parse_directory(project_animus_skills):
                    self._skills[skill.name] = skill

        return self._skills

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name.

        Args:
            name: Skill name

        Returns:
            Skill if found, None otherwise
        """
        return self._skills.get(name)

    def list(self) -> list[Skill]:
        """List all discovered skills.

        Returns:
            List of all skills
        """
        return list(self._skills.values())

    def list_enabled(self) -> list[Skill]:
        """List only enabled skills.

        Returns:
            List of enabled skills
        """
        return [s for s in self._skills.values() if s.metadata.enabled]

    def register(self, skill: Skill) -> None:
        """Register a skill manually.

        Args:
            skill: Skill to register
        """
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> bool:
        """Unregister a skill.

        Args:
            name: Skill name to remove

        Returns:
            True if skill was removed, False if not found
        """
        if name in self._skills:
            del self._skills[name]
            return True
        return False

    def search(self, query: str) -> list[Skill]:
        """Search skills by name or description.

        Args:
            query: Search query

        Returns:
            List of matching skills
        """
        query = query.lower()
        results = []

        for skill in self._skills.values():
            # Search in name
            if query in skill.name.lower():
                results.append(skill)
                continue

            # Search in description
            if query in skill.description.lower():
                results.append(skill)
                continue

            # Search in tags
            for tag in skill.metadata.tags:
                if query in tag.lower():
                    results.append(skill)
                    break

        return results

    def get_by_tag(self, tag: str) -> list[Skill]:
        """Get all skills with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of skills with the tag
        """
        return [
            skill for skill in self._skills.values()
            if tag in skill.metadata.tags
        ]

    def install_from_url(self, url: str, target_dir: Optional[Path] = None) -> Skill:
        """Install a skill from a URL (GitHub, etc.).

        Args:
            url: URL to skill repository or SKILL.md file
            target_dir: Directory to install to (default: user skills dir)

        Returns:
            Installed skill

        Raises:
            ValueError: If installation fails
        """
        target = target_dir or self.user_skills_dir
        target.mkdir(parents=True, exist_ok=True)

        # Handle GitHub URLs
        if "github.com" in url:
            return self._install_from_github(url, target)

        # Handle direct SKILL.md URLs
        if url.endswith(".md"):
            return self._install_from_raw_url(url, target)

        raise ValueError(f"Unsupported URL format: {url}")

    def _install_from_github(self, url: str, target: Path) -> Skill:
        """Install skill from GitHub repository."""
        import urllib.request
        import json

        # Parse GitHub URL
        # Format: https://github.com/owner/repo or https://github.com/owner/repo/tree/branch/path
        parts = url.replace("https://github.com/", "").split("/")

        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub URL: {url}")

        owner, repo = parts[0], parts[1]

        # Determine the path to SKILL.md
        if len(parts) > 4 and parts[2] == "tree":
            # URL includes branch and path
            branch = parts[3]
            skill_path = "/".join(parts[4:])
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{skill_path}/SKILL.md"
        else:
            # Default to main branch, root SKILL.md
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/SKILL.md"

        try:
            with urllib.request.urlopen(raw_url) as response:
                content = response.read().decode("utf-8")
        except Exception as e:
            raise ValueError(f"Failed to fetch SKILL.md from {raw_url}: {e}")

        # Parse the skill
        skill = self._parser.parse_string(content)

        # Create skill directory and save
        skill_dir = target / skill.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(content, encoding="utf-8")

        # Update source path and register
        skill.source_path = skill_file
        self._skills[skill.name] = skill

        return skill

    def _install_from_raw_url(self, url: str, target: Path) -> Skill:
        """Install skill from raw URL to SKILL.md."""
        import urllib.request

        try:
            with urllib.request.urlopen(url) as response:
                content = response.read().decode("utf-8")
        except Exception as e:
            raise ValueError(f"Failed to fetch SKILL.md from {url}: {e}")

        # Parse the skill
        skill = self._parser.parse_string(content)

        # Create skill directory and save
        skill_dir = target / skill.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(content, encoding="utf-8")

        # Update source path and register
        skill.source_path = skill_file
        self._skills[skill.name] = skill

        return skill

    def create(self, name: str, description: str, target_dir: Optional[Path] = None) -> Path:
        """Create a new skill from template.

        Args:
            name: Skill name (lowercase, hyphen-separated)
            description: Brief skill description
            target_dir: Directory to create in (default: user skills dir)

        Returns:
            Path to created SKILL.md file
        """
        from src.skills.parser import create_skill_template

        target = target_dir or self.user_skills_dir
        target.mkdir(parents=True, exist_ok=True)

        # Normalize name
        name = name.lower().replace(" ", "-").replace("_", "-")

        # Create skill directory
        skill_dir = target / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Create SKILL.md
        skill_file = skill_dir / "SKILL.md"
        content = create_skill_template(name, description)
        skill_file.write_text(content, encoding="utf-8")

        # Parse and register
        skill = self._parser.parse_file(skill_file)
        self._skills[skill.name] = skill

        return skill_file
