"""BuilderQuery interface for run analysis and self-improvement.

This module provides tools for analyzing agent runs, identifying patterns,
and generating actionable improvement suggestions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from src.core.decision import DecisionType, OutcomeStatus
from src.core.run import Run, RunStatus, RunStore


class SuggestionPriority(Enum):
    """Priority level for improvement suggestions."""
    CRITICAL = "critical"    # Must address immediately
    HIGH = "high"           # Should address soon
    MEDIUM = "medium"       # Worth addressing
    LOW = "low"             # Nice to have


class SuggestionCategory(Enum):
    """Category of improvement suggestion."""
    ERROR_PATTERN = "error_pattern"         # Recurring errors
    PERFORMANCE = "performance"             # Speed/efficiency issues
    TOOL_USAGE = "tool_usage"               # Tool selection/usage problems
    STRATEGY = "strategy"                   # High-level approach issues
    RESOURCE = "resource"                   # Token/resource usage
    SUCCESS_RATE = "success_rate"           # Overall success patterns


@dataclass
class Suggestion:
    """An actionable improvement suggestion."""
    id: str
    category: SuggestionCategory
    priority: SuggestionPriority
    title: str
    description: str
    evidence: list[str] = field(default_factory=list)  # Supporting data points
    affected_runs: list[str] = field(default_factory=list)  # Run IDs
    suggested_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "category": self.category.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "affected_runs": self.affected_runs,
            "suggested_actions": self.suggested_actions,
            "metadata": self.metadata,
        }


@dataclass
class AnalysisResult:
    """Results of analyzing runs."""
    total_runs: int
    analyzed_runs: int
    date_range: tuple[Optional[datetime], Optional[datetime]]
    suggestions: list[Suggestion]
    patterns: dict[str, Any]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_runs": self.total_runs,
            "analyzed_runs": self.analyzed_runs,
            "date_range": [
                self.date_range[0].isoformat() if self.date_range[0] else None,
                self.date_range[1].isoformat() if self.date_range[1] else None,
            ],
            "suggestions": [s.to_dict() for s in self.suggestions],
            "patterns": self.patterns,
            "summary": self.summary,
        }


class BuilderQuery:
    """Analyzes runs and generates improvement suggestions.

    The BuilderQuery provides tools for introspection and self-improvement
    by analyzing patterns in past runs, identifying recurring issues,
    and suggesting concrete improvements.
    """

    def __init__(self, store: Optional[RunStore] = None):
        """Initialize BuilderQuery.

        Args:
            store: RunStore instance. If None, creates default store.
        """
        self.store = store or RunStore()
        self._suggestion_counter = 0

    def _generate_suggestion_id(self) -> str:
        """Generate unique suggestion ID."""
        self._suggestion_counter += 1
        return f"sug_{self._suggestion_counter:04d}"

    def analyze(
        self,
        goal_filter: Optional[str] = None,
        days: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> AnalysisResult:
        """Analyze runs and generate suggestions.

        Args:
            goal_filter: Filter runs by goal substring.
            days: Only analyze runs from last N days.
            limit: Maximum number of runs to analyze.

        Returns:
            AnalysisResult with suggestions and patterns.
        """
        # Gather runs
        runs = self._gather_runs(goal_filter, days, limit)

        if not runs:
            return AnalysisResult(
                total_runs=0,
                analyzed_runs=0,
                date_range=(None, None),
                suggestions=[],
                patterns={},
                summary="No runs found matching criteria.",
            )

        # Analyze patterns
        patterns = self._analyze_patterns(runs)

        # Generate suggestions
        suggestions = []
        suggestions.extend(self._analyze_error_patterns(runs, patterns))
        suggestions.extend(self._analyze_performance(runs, patterns))
        suggestions.extend(self._analyze_tool_usage(runs, patterns))
        suggestions.extend(self._analyze_success_rates(runs, patterns))

        # Sort by priority
        priority_order = {
            SuggestionPriority.CRITICAL: 0,
            SuggestionPriority.HIGH: 1,
            SuggestionPriority.MEDIUM: 2,
            SuggestionPriority.LOW: 3,
        }
        suggestions.sort(key=lambda s: priority_order[s.priority])

        # Calculate date range
        dates = [r.started_at for r in runs if r.started_at]
        date_range = (min(dates), max(dates)) if dates else (None, None)

        # Generate summary
        summary = self._generate_summary(runs, suggestions, patterns)

        return AnalysisResult(
            total_runs=len(list(self.store.iter_runs())),
            analyzed_runs=len(runs),
            date_range=date_range,
            suggestions=suggestions,
            patterns=patterns,
            summary=summary,
        )

    def _gather_runs(
        self,
        goal_filter: Optional[str],
        days: Optional[int],
        limit: Optional[int],
    ) -> list[Run]:
        """Gather runs matching criteria."""
        runs = []

        if goal_filter:
            runs = self.store.find_by_goal(goal_filter)
        else:
            runs = list(self.store.iter_runs())

        if days:
            cutoff = datetime.now() - timedelta(days=days)
            runs = [r for r in runs if r.started_at and r.started_at >= cutoff]

        # Sort by date (newest first)
        runs.sort(key=lambda r: r.started_at or datetime.min, reverse=True)

        if limit:
            runs = runs[:limit]

        return runs

    def _analyze_patterns(self, runs: list[Run]) -> dict[str, Any]:
        """Extract patterns from runs."""
        patterns: dict[str, Any] = {
            "status_distribution": Counter(),
            "error_frequency": Counter(),
            "tool_success_rates": defaultdict(lambda: {"success": 0, "total": 0}),
            "decision_types": Counter(),
            "avg_tokens": 0,
            "avg_duration_ms": 0,
            "avg_turns": 0,
            "total_tool_calls": 0,
            "total_decisions": 0,
        }

        total_tokens = 0
        total_duration = 0.0
        total_turns = 0

        for run in runs:
            # Status distribution
            patterns["status_distribution"][run.status.value] += 1

            # Error patterns
            for error in run.metrics.errors:
                # Extract error type/category
                error_type = self._categorize_error(error)
                patterns["error_frequency"][error_type] += 1

            # Tool usage
            patterns["total_tool_calls"] += run.metrics.tool_calls

            # Decision types
            for decision in run.decisions:
                patterns["decision_types"][decision.decision_type.value] += 1
            patterns["total_decisions"] += run.metrics.decisions_made

            # Aggregates
            total_tokens += run.metrics.tokens_used
            total_duration += run.duration_ms
            total_turns += run.metrics.turn_count

        n = len(runs)
        patterns["avg_tokens"] = total_tokens / n if n > 0 else 0
        patterns["avg_duration_ms"] = total_duration / n if n > 0 else 0
        patterns["avg_turns"] = total_turns / n if n > 0 else 0

        return patterns

    def _categorize_error(self, error: str) -> str:
        """Categorize an error message into a type."""
        error_lower = error.lower()

        if "timeout" in error_lower:
            return "timeout"
        elif "rate limit" in error_lower or "429" in error:
            return "rate_limit"
        elif "auth" in error_lower or "401" in error or "403" in error:
            return "authentication"
        elif "not found" in error_lower or "404" in error:
            return "not_found"
        elif "permission" in error_lower or "denied" in error_lower:
            return "permission_denied"
        elif "memory" in error_lower or "oom" in error_lower:
            return "memory"
        elif "connection" in error_lower or "network" in error_lower:
            return "network"
        elif "syntax" in error_lower or "parse" in error_lower:
            return "syntax"
        elif "import" in error_lower or "module" in error_lower:
            return "import"
        else:
            return "other"

    def _analyze_error_patterns(
        self,
        runs: list[Run],
        patterns: dict[str, Any],
    ) -> list[Suggestion]:
        """Generate suggestions from error patterns."""
        suggestions = []
        error_freq = patterns["error_frequency"]

        # Check for recurring errors
        for error_type, count in error_freq.most_common(5):
            if count < 2:
                continue

            affected = [r.id for r in runs if any(
                self._categorize_error(e) == error_type
                for e in r.metrics.errors
            )]

            if error_type == "timeout":
                suggestions.append(Suggestion(
                    id=self._generate_suggestion_id(),
                    category=SuggestionCategory.ERROR_PATTERN,
                    priority=SuggestionPriority.HIGH,
                    title=f"Recurring timeout errors ({count} occurrences)",
                    description="Multiple runs are failing due to timeout errors. "
                               "This may indicate operations are taking too long.",
                    evidence=[f"{count} timeout errors across {len(affected)} runs"],
                    affected_runs=affected,
                    suggested_actions=[
                        "Increase timeout limits for long-running operations",
                        "Break large operations into smaller chunks",
                        "Add progress indicators for long tasks",
                    ],
                ))
            elif error_type == "rate_limit":
                suggestions.append(Suggestion(
                    id=self._generate_suggestion_id(),
                    category=SuggestionCategory.ERROR_PATTERN,
                    priority=SuggestionPriority.HIGH,
                    title=f"Rate limiting detected ({count} occurrences)",
                    description="API rate limits are being hit frequently. "
                               "Consider adding backoff strategies.",
                    evidence=[f"{count} rate limit errors across {len(affected)} runs"],
                    affected_runs=affected,
                    suggested_actions=[
                        "Implement exponential backoff for API calls",
                        "Add request caching to reduce API calls",
                        "Consider batching requests where possible",
                    ],
                ))
            elif error_type == "permission_denied":
                suggestions.append(Suggestion(
                    id=self._generate_suggestion_id(),
                    category=SuggestionCategory.ERROR_PATTERN,
                    priority=SuggestionPriority.MEDIUM,
                    title=f"Permission errors ({count} occurrences)",
                    description="Operations are failing due to permission issues. "
                               "Review file and command permissions.",
                    evidence=[f"{count} permission errors across {len(affected)} runs"],
                    affected_runs=affected,
                    suggested_actions=[
                        "Review permission settings for accessed resources",
                        "Add better error handling for permission failures",
                        "Consider prompting user for elevated permissions",
                    ],
                ))
            elif error_type == "network":
                suggestions.append(Suggestion(
                    id=self._generate_suggestion_id(),
                    category=SuggestionCategory.ERROR_PATTERN,
                    priority=SuggestionPriority.MEDIUM,
                    title=f"Network connectivity issues ({count} occurrences)",
                    description="Network-related errors are occurring. "
                               "Add retry logic and better offline handling.",
                    evidence=[f"{count} network errors across {len(affected)} runs"],
                    affected_runs=affected,
                    suggested_actions=[
                        "Add automatic retry with exponential backoff",
                        "Implement connection health checks",
                        "Add graceful degradation for offline mode",
                    ],
                ))

        return suggestions

    def _analyze_performance(
        self,
        runs: list[Run],
        patterns: dict[str, Any],
    ) -> list[Suggestion]:
        """Generate performance-related suggestions."""
        suggestions = []

        # High token usage
        avg_tokens = patterns["avg_tokens"]
        if avg_tokens > 50000:
            high_token_runs = [
                r.id for r in runs if r.metrics.tokens_used > avg_tokens * 1.5
            ]
            suggestions.append(Suggestion(
                id=self._generate_suggestion_id(),
                category=SuggestionCategory.RESOURCE,
                priority=SuggestionPriority.MEDIUM,
                title=f"High token consumption (avg {avg_tokens:.0f} tokens)",
                description="Token usage is high. Consider optimizing prompts "
                           "and context management.",
                evidence=[
                    f"Average: {avg_tokens:.0f} tokens per run",
                    f"{len(high_token_runs)} runs with exceptionally high usage",
                ],
                affected_runs=high_token_runs,
                suggested_actions=[
                    "Enable more aggressive context compaction",
                    "Use shorter system prompts",
                    "Summarize long outputs before adding to context",
                ],
            ))

        # Slow runs
        avg_duration = patterns["avg_duration_ms"]
        if avg_duration > 60000:  # > 1 minute average
            slow_runs = [
                r.id for r in runs if r.duration_ms > avg_duration * 1.5
            ]
            suggestions.append(Suggestion(
                id=self._generate_suggestion_id(),
                category=SuggestionCategory.PERFORMANCE,
                priority=SuggestionPriority.MEDIUM,
                title=f"Slow run execution (avg {avg_duration/1000:.1f}s)",
                description="Runs are taking longer than expected. "
                           "Consider optimizing operations.",
                evidence=[
                    f"Average duration: {avg_duration/1000:.1f} seconds",
                    f"{len(slow_runs)} runs significantly slower than average",
                ],
                affected_runs=slow_runs,
                suggested_actions=[
                    "Profile slow operations to identify bottlenecks",
                    "Cache expensive computations",
                    "Consider parallelizing independent operations",
                ],
            ))

        # Many turns
        avg_turns = patterns["avg_turns"]
        if avg_turns > 10:
            many_turn_runs = [
                r.id for r in runs if r.metrics.turn_count > avg_turns * 1.5
            ]
            suggestions.append(Suggestion(
                id=self._generate_suggestion_id(),
                category=SuggestionCategory.STRATEGY,
                priority=SuggestionPriority.LOW,
                title=f"High turn count (avg {avg_turns:.1f} turns)",
                description="Runs are requiring many turns to complete. "
                           "Consider improving planning efficiency.",
                evidence=[
                    f"Average: {avg_turns:.1f} turns per run",
                    f"{len(many_turn_runs)} runs with excessive turns",
                ],
                affected_runs=many_turn_runs,
                suggested_actions=[
                    "Improve initial planning to reduce iterations",
                    "Use more efficient tool combinations",
                    "Add better success criteria to avoid unnecessary steps",
                ],
            ))

        return suggestions

    def _analyze_tool_usage(
        self,
        runs: list[Run],
        patterns: dict[str, Any],
    ) -> list[Suggestion]:
        """Generate tool usage suggestions."""
        suggestions = []

        # Calculate overall tool success rate
        total_calls = sum(r.metrics.tool_calls for r in runs)
        total_successes = sum(r.metrics.tool_successes for r in runs)

        if total_calls > 0:
            success_rate = total_successes / total_calls

            if success_rate < 0.8:
                low_success_runs = [
                    r.id for r in runs
                    if r.metrics.tool_calls > 0 and
                    r.metrics.tool_successes / r.metrics.tool_calls < 0.7
                ]
                suggestions.append(Suggestion(
                    id=self._generate_suggestion_id(),
                    category=SuggestionCategory.TOOL_USAGE,
                    priority=SuggestionPriority.HIGH,
                    title=f"Low tool success rate ({success_rate*100:.1f}%)",
                    description="Tool calls are frequently failing. "
                               "Review tool usage patterns and error handling.",
                    evidence=[
                        f"Overall success rate: {success_rate*100:.1f}%",
                        f"Total calls: {total_calls}, successes: {total_successes}",
                        f"{len(low_success_runs)} runs with particularly low success",
                    ],
                    affected_runs=low_success_runs,
                    suggested_actions=[
                        "Add input validation before tool calls",
                        "Improve error recovery strategies",
                        "Review common failure patterns",
                    ],
                ))

        return suggestions

    def _analyze_success_rates(
        self,
        runs: list[Run],
        patterns: dict[str, Any],
    ) -> list[Suggestion]:
        """Analyze overall success rates."""
        suggestions = []
        status_dist = patterns["status_distribution"]

        completed = status_dist.get("completed", 0)
        failed = status_dist.get("failed", 0)
        total = completed + failed

        if total > 0:
            failure_rate = failed / total

            if failure_rate > 0.3:
                failed_runs = [
                    r.id for r in runs if r.status == RunStatus.FAILED
                ]
                suggestions.append(Suggestion(
                    id=self._generate_suggestion_id(),
                    category=SuggestionCategory.SUCCESS_RATE,
                    priority=SuggestionPriority.CRITICAL,
                    title=f"High failure rate ({failure_rate*100:.1f}%)",
                    description="A significant portion of runs are failing. "
                               "This requires immediate attention.",
                    evidence=[
                        f"Completed: {completed}, Failed: {failed}",
                        f"Failure rate: {failure_rate*100:.1f}%",
                    ],
                    affected_runs=failed_runs,
                    suggested_actions=[
                        "Review failed runs for common causes",
                        "Add better error handling and recovery",
                        "Consider adding pre-flight checks",
                        "Implement graceful degradation",
                    ],
                ))

        return suggestions

    def _generate_summary(
        self,
        runs: list[Run],
        suggestions: list[Suggestion],
        patterns: dict[str, Any],
    ) -> str:
        """Generate a human-readable summary."""
        lines = []

        # Overview
        status_dist = patterns["status_distribution"]
        completed = status_dist.get("completed", 0)
        failed = status_dist.get("failed", 0)

        lines.append(f"Analyzed {len(runs)} runs:")
        lines.append(f"  - Completed: {completed}")
        lines.append(f"  - Failed: {failed}")
        lines.append(f"  - Other: {len(runs) - completed - failed}")
        lines.append("")

        # Key metrics
        lines.append("Key Metrics:")
        lines.append(f"  - Avg tokens: {patterns['avg_tokens']:.0f}")
        lines.append(f"  - Avg duration: {patterns['avg_duration_ms']/1000:.1f}s")
        lines.append(f"  - Avg turns: {patterns['avg_turns']:.1f}")
        lines.append("")

        # Suggestions summary
        if suggestions:
            critical = sum(1 for s in suggestions if s.priority == SuggestionPriority.CRITICAL)
            high = sum(1 for s in suggestions if s.priority == SuggestionPriority.HIGH)

            lines.append(f"Found {len(suggestions)} suggestions:")
            if critical:
                lines.append(f"  - {critical} CRITICAL")
            if high:
                lines.append(f"  - {high} HIGH priority")
        else:
            lines.append("No significant issues found.")

        return "\n".join(lines)

    def get_run_details(self, run_id: str) -> Optional[dict[str, Any]]:
        """Get detailed analysis of a specific run.

        Args:
            run_id: The run ID to analyze.

        Returns:
            Detailed analysis dict or None if not found.
        """
        run = self.store.load(run_id)
        if not run:
            return None

        # Analyze decisions
        decision_analysis = []
        for decision in run.decisions:
            decision_analysis.append({
                "type": decision.decision_type.value,
                "intent": decision.intent,
                "options_count": len(decision.options),
                "chosen": decision.chosen_option.description if decision.chosen_option else None,
                "reasoning": decision.reasoning,
            })

        return {
            "id": run.id,
            "goal": run.goal,
            "status": run.status.value,
            "duration_ms": run.duration_ms,
            "metrics": run.metrics.to_dict(),
            "decision_count": len(run.decisions),
            "decisions": decision_analysis,
            "errors": run.metrics.errors,
            "success_rate": run.metrics.success_rate,
        }

    def compare_runs(
        self,
        run_ids: list[str],
    ) -> dict[str, Any]:
        """Compare multiple runs to identify differences.

        Args:
            run_ids: List of run IDs to compare.

        Returns:
            Comparison analysis.
        """
        runs = []
        for run_id in run_ids:
            run = self.store.load(run_id)
            if run:
                runs.append(run)

        if len(runs) < 2:
            return {"error": "Need at least 2 valid runs to compare"}

        comparison = {
            "runs": [
                {
                    "id": r.id,
                    "goal": r.goal[:50] + "..." if len(r.goal) > 50 else r.goal,
                    "status": r.status.value,
                    "tokens": r.metrics.tokens_used,
                    "duration_ms": r.duration_ms,
                    "turns": r.metrics.turn_count,
                    "tool_calls": r.metrics.tool_calls,
                    "success_rate": r.metrics.success_rate,
                }
                for r in runs
            ],
            "analysis": {},
        }

        # Identify best/worst
        by_success_rate = sorted(runs, key=lambda r: r.metrics.success_rate, reverse=True)
        by_tokens = sorted(runs, key=lambda r: r.metrics.tokens_used)
        by_duration = sorted(runs, key=lambda r: r.duration_ms)

        comparison["analysis"]["most_efficient"] = {
            "by_success_rate": by_success_rate[0].id,
            "by_tokens": by_tokens[0].id,
            "by_duration": by_duration[0].id,
        }

        return comparison

    def get_trends(self, days: int = 30) -> dict[str, Any]:
        """Analyze trends over time.

        Args:
            days: Number of days to analyze.

        Returns:
            Trend analysis.
        """
        cutoff = datetime.now() - timedelta(days=days)
        runs = [
            r for r in self.store.iter_runs()
            if r.started_at and r.started_at >= cutoff
        ]

        if not runs:
            return {"error": "No runs in the specified period"}

        # Group by day
        by_day: dict[str, list[Run]] = defaultdict(list)
        for run in runs:
            if run.started_at:
                day_key = run.started_at.strftime("%Y-%m-%d")
                by_day[day_key].append(run)

        daily_stats = []
        for day in sorted(by_day.keys()):
            day_runs = by_day[day]
            completed = sum(1 for r in day_runs if r.status == RunStatus.COMPLETED)
            failed = sum(1 for r in day_runs if r.status == RunStatus.FAILED)
            total_tokens = sum(r.metrics.tokens_used for r in day_runs)

            daily_stats.append({
                "date": day,
                "runs": len(day_runs),
                "completed": completed,
                "failed": failed,
                "success_rate": completed / (completed + failed) if (completed + failed) > 0 else 0,
                "avg_tokens": total_tokens / len(day_runs) if day_runs else 0,
            })

        # Calculate overall trends
        if len(daily_stats) >= 2:
            first_half = daily_stats[:len(daily_stats)//2]
            second_half = daily_stats[len(daily_stats)//2:]

            first_rate = sum(d["success_rate"] for d in first_half) / len(first_half)
            second_rate = sum(d["success_rate"] for d in second_half) / len(second_half)

            trend = "improving" if second_rate > first_rate else "declining" if second_rate < first_rate else "stable"
        else:
            trend = "insufficient_data"

        return {
            "period_days": days,
            "total_runs": len(runs),
            "daily_stats": daily_stats,
            "overall_trend": trend,
        }
