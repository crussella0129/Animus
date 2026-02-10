"""Auth profile rotation — failover across multiple API credentials.

Manages multiple API key profiles with cooldown tracking. When an auth
failure occurs, the rotator switches to the next available profile.

Usage:
    rotator = AuthRotator([
        AuthProfile("primary", api_key="sk-aaa"),
        AuthProfile("backup", api_key="sk-bbb", api_base="https://alt.api/v1"),
    ])

    profile = rotator.current          # AuthProfile "primary"
    rotator.record_failure("primary")  # marks cooldown, switches to "backup"
    profile = rotator.current          # AuthProfile "backup"
    rotator.record_success("backup")   # tracks usage
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AuthProfile:
    """A single API credential profile."""

    name: str
    api_key: str
    api_base: Optional[str] = None  # Override API base URL (None = provider default)
    models: Optional[list[str]] = None  # Models this key supports (None = all)

    # Cooldown after auth failure
    cooldown_seconds: float = 300.0  # 5 minutes default

    # Runtime state
    consecutive_failures: int = field(default=0, repr=False)
    last_failure_time: float = field(default=0.0, repr=False)
    total_failures: int = field(default=0, repr=False)
    total_successes: int = field(default=0, repr=False)
    total_requests: int = field(default=0, repr=False)

    @property
    def is_on_cooldown(self) -> bool:
        """Check if this profile is currently on cooldown."""
        if self.last_failure_time == 0.0:
            return False
        elapsed = time.monotonic() - self.last_failure_time
        return elapsed < self.cooldown_seconds


@dataclass
class RotationEvent:
    """Record of a profile rotation."""

    timestamp: float
    from_profile: str
    to_profile: str
    reason: str  # "auth_failure", "cooldown_expired", "manual"


class AuthRotator:
    """Manages multiple auth profiles with automatic failover.

    Profiles are tried in order. When the current profile hits an auth
    failure, it's placed on cooldown and the next available profile is
    selected. Profiles come off cooldown automatically.
    """

    def __init__(self, profiles: list[AuthProfile]):
        """Initialize the auth rotator.

        Args:
            profiles: Ordered list of auth profiles (preferred first).

        Raises:
            ValueError: If no profiles are provided.
        """
        if not profiles:
            raise ValueError("AuthRotator requires at least one profile")

        self._profiles = list(profiles)
        self._profile_map = {p.name: p for p in self._profiles}
        self._current_index = 0
        self._events: list[RotationEvent] = []

    @property
    def current(self) -> AuthProfile:
        """Get the currently active profile."""
        return self._profiles[self._current_index]

    @property
    def profiles(self) -> list[AuthProfile]:
        """Get all profiles."""
        return list(self._profiles)

    @property
    def events(self) -> list[RotationEvent]:
        """Get rotation event history."""
        return list(self._events)

    def get_profile(self, name: str) -> Optional[AuthProfile]:
        """Get a profile by name."""
        return self._profile_map.get(name)

    def record_failure(self, profile_name: Optional[str] = None) -> bool:
        """Record an auth failure for a profile.

        Args:
            profile_name: Profile that failed (None = current).

        Returns:
            True if rotation to a new profile occurred.
        """
        profile = self._resolve_profile(profile_name)
        profile.consecutive_failures += 1
        profile.total_failures += 1
        profile.last_failure_time = time.monotonic()

        logger.warning(
            "Auth failure for profile '%s' (failures: %d)",
            profile.name,
            profile.consecutive_failures,
        )

        # Try to rotate to next available profile
        return self._rotate_to_next(reason="auth_failure")

    def record_success(self, profile_name: Optional[str] = None) -> None:
        """Record a successful request for a profile.

        Args:
            profile_name: Profile that succeeded (None = current).
        """
        profile = self._resolve_profile(profile_name)
        profile.consecutive_failures = 0
        profile.total_successes += 1
        profile.total_requests += 1

    def record_request(self, profile_name: Optional[str] = None) -> None:
        """Record a request attempt (before success/failure is known).

        Args:
            profile_name: Profile used (None = current).
        """
        profile = self._resolve_profile(profile_name)
        profile.total_requests += 1

    def _resolve_profile(self, name: Optional[str]) -> AuthProfile:
        """Resolve a profile name to a profile object."""
        if name is None:
            return self.current
        profile = self._profile_map.get(name)
        if profile is None:
            raise KeyError(f"Unknown profile: {name}")
        return profile

    def _rotate_to_next(self, reason: str) -> bool:
        """Find and switch to the next available (non-cooldown) profile.

        Returns:
            True if a rotation occurred.
        """
        old_name = self.current.name
        n = len(self._profiles)

        for offset in range(1, n):
            candidate_idx = (self._current_index + offset) % n
            candidate = self._profiles[candidate_idx]

            if not candidate.is_on_cooldown:
                self._current_index = candidate_idx
                event = RotationEvent(
                    timestamp=time.monotonic(),
                    from_profile=old_name,
                    to_profile=candidate.name,
                    reason=reason,
                )
                self._events.append(event)
                logger.info(
                    "Auth rotation: '%s' → '%s' (%s)",
                    old_name,
                    candidate.name,
                    reason,
                )
                return True

        logger.warning(
            "All auth profiles on cooldown — staying on '%s'",
            old_name,
        )
        return False

    def try_recover_preferred(self) -> bool:
        """Try to return to the preferred (first) profile if its cooldown expired.

        Returns:
            True if recovery occurred.
        """
        if self._current_index == 0:
            return False

        preferred = self._profiles[0]
        if not preferred.is_on_cooldown:
            old_name = self.current.name
            self._current_index = 0
            preferred.consecutive_failures = 0

            event = RotationEvent(
                timestamp=time.monotonic(),
                from_profile=old_name,
                to_profile=preferred.name,
                reason="cooldown_expired",
            )
            self._events.append(event)
            logger.info(
                "Auth recovery: '%s' → '%s' (cooldown expired)",
                old_name,
                preferred.name,
            )
            return True

        return False

    def reset(self) -> None:
        """Reset all profiles to initial state."""
        self._current_index = 0
        self._events.clear()
        for p in self._profiles:
            p.consecutive_failures = 0
            p.last_failure_time = 0.0
            p.total_failures = 0
            p.total_successes = 0
            p.total_requests = 0

    def stats(self) -> dict:
        """Get statistics for all profiles."""
        return {
            "current_profile": self.current.name,
            "rotation_count": len(self._events),
            "profiles": [
                {
                    "name": p.name,
                    "on_cooldown": p.is_on_cooldown,
                    "consecutive_failures": p.consecutive_failures,
                    "total_failures": p.total_failures,
                    "total_successes": p.total_successes,
                    "total_requests": p.total_requests,
                }
                for p in self._profiles
            ],
        }
