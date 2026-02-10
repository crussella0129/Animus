"""Lane-based command queueing — serialize concurrent agent operations.

Prevents interleaving of concurrent runs by assigning each session a
lane. Commands within a lane execute sequentially. Multiple lanes can
run in parallel. Supports priority ordering and pause/resume.

Usage:
    queue = CommandQueue(max_lanes=4)

    # Submit work to a lane
    result = await queue.submit("session-1", my_coroutine())

    # Priority submission (higher = sooner)
    result = await queue.submit("session-1", important_work(), priority=10)

    # Pause/resume a lane
    queue.pause_lane("session-1")
    queue.resume_lane("session-1")

    # Shutdown
    await queue.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


class LaneStatus(str, Enum):
    """Status of a lane."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    DRAINING = "draining"  # Finishing current work, rejecting new


@dataclass
class QueueEntry:
    """An entry in the command queue."""

    id: int
    lane_id: str
    coroutine: Awaitable[Any]
    priority: int = 0  # Higher = processed sooner
    submitted_at: float = field(default_factory=time.monotonic)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None

    def __lt__(self, other: QueueEntry) -> bool:
        """Higher priority first, then FIFO by id."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.id < other.id


@dataclass
class LaneStats:
    """Statistics for a single lane."""

    lane_id: str
    status: LaneStatus
    queued: int
    completed: int
    failed: int
    total_wait_time: float
    total_run_time: float


class Lane:
    """A single execution lane — processes entries sequentially."""

    def __init__(self, lane_id: str):
        self.lane_id = lane_id
        self.status = LaneStatus.IDLE
        self._queue: asyncio.PriorityQueue[QueueEntry] = asyncio.PriorityQueue()
        self._worker_task: Optional[asyncio.Task] = None
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially
        self._completed = 0
        self._failed = 0
        self._total_wait_time = 0.0
        self._total_run_time = 0.0

    async def enqueue(self, entry: QueueEntry) -> None:
        """Add an entry to the lane's queue."""
        await self._queue.put(entry)

        # Start worker if not running
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())

    async def _worker(self) -> None:
        """Process entries sequentially."""
        self.status = LaneStatus.RUNNING

        while not self._queue.empty():
            # Wait if paused
            await self._pause_event.wait()

            try:
                entry = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            entry.started_at = time.monotonic()
            wait_time = entry.started_at - entry.submitted_at
            self._total_wait_time += wait_time

            try:
                entry.result = await entry.coroutine
                entry.completed_at = time.monotonic()
                self._completed += 1
            except Exception as e:
                entry.error = e
                entry.completed_at = time.monotonic()
                self._failed += 1
                logger.warning(
                    "Lane '%s' entry %d failed: %s",
                    self.lane_id, entry.id, e,
                )

            run_time = entry.completed_at - entry.started_at
            self._total_run_time += run_time
            self._queue.task_done()

        self.status = LaneStatus.IDLE

    def pause(self) -> None:
        """Pause processing (current entry finishes, then waits)."""
        self._pause_event.clear()
        if self.status == LaneStatus.RUNNING:
            self.status = LaneStatus.PAUSED
        logger.info("Lane '%s' paused", self.lane_id)

    def resume(self) -> None:
        """Resume processing."""
        self._pause_event.set()
        if self.status == LaneStatus.PAUSED:
            self.status = LaneStatus.RUNNING
        logger.info("Lane '%s' resumed", self.lane_id)

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    @property
    def queued_count(self) -> int:
        return self._queue.qsize()

    def stats(self) -> LaneStats:
        return LaneStats(
            lane_id=self.lane_id,
            status=self.status,
            queued=self._queue.qsize(),
            completed=self._completed,
            failed=self._failed,
            total_wait_time=self._total_wait_time,
            total_run_time=self._total_run_time,
        )


class CommandQueue:
    """Multi-lane command queue with priority and pause/resume.

    Each lane processes commands sequentially. Different lanes
    run in parallel. Sessions are assigned to lanes by their ID.
    """

    def __init__(self, max_lanes: int = 4):
        """Initialize the command queue.

        Args:
            max_lanes: Maximum concurrent lanes.
        """
        self._max_lanes = max_lanes
        self._lanes: dict[str, Lane] = {}
        self._next_id = 0
        self._shutdown = False

    def _get_or_create_lane(self, lane_id: str) -> Lane:
        """Get or create a lane."""
        if lane_id not in self._lanes:
            if len(self._lanes) >= self._max_lanes:
                # Find an idle lane to reuse
                for lid, lane in self._lanes.items():
                    if lane.status == LaneStatus.IDLE and lane.queued_count == 0:
                        del self._lanes[lid]
                        break
                else:
                    raise RuntimeError(
                        f"Maximum lanes ({self._max_lanes}) reached. "
                        f"Active lanes: {list(self._lanes.keys())}"
                    )
            self._lanes[lane_id] = Lane(lane_id)
        return self._lanes[lane_id]

    async def submit(
        self,
        lane_id: str,
        coroutine: Awaitable[Any],
        priority: int = 0,
    ) -> QueueEntry:
        """Submit a coroutine to a lane for sequential execution.

        Args:
            lane_id: Lane/session identifier.
            coroutine: The async work to execute.
            priority: Higher priority = processed sooner.

        Returns:
            QueueEntry that will be populated with result/error.
        """
        if self._shutdown:
            raise RuntimeError("Queue is shut down")

        self._next_id += 1
        entry = QueueEntry(
            id=self._next_id,
            lane_id=lane_id,
            coroutine=coroutine,
            priority=priority,
        )

        lane = self._get_or_create_lane(lane_id)
        await lane.enqueue(entry)

        # Wait for completion
        while entry.completed_at is None:
            await asyncio.sleep(0.01)

        return entry

    def pause_lane(self, lane_id: str) -> bool:
        """Pause a lane. Returns True if the lane exists."""
        lane = self._lanes.get(lane_id)
        if lane:
            lane.pause()
            return True
        return False

    def resume_lane(self, lane_id: str) -> bool:
        """Resume a paused lane. Returns True if the lane exists."""
        lane = self._lanes.get(lane_id)
        if lane:
            lane.resume()
            return True
        return False

    def get_lane_status(self, lane_id: str) -> Optional[LaneStatus]:
        """Get the status of a lane."""
        lane = self._lanes.get(lane_id)
        return lane.status if lane else None

    @property
    def active_lanes(self) -> int:
        """Count of lanes with queued or running work."""
        return sum(
            1 for lane in self._lanes.values()
            if lane.status != LaneStatus.IDLE or lane.queued_count > 0
        )

    async def shutdown(self, timeout: float = 10.0) -> None:
        """Stop accepting new work and wait for current work to complete."""
        self._shutdown = True

        # Wait for all lanes to finish
        deadline = time.monotonic() + timeout
        for lane in self._lanes.values():
            lane.resume()  # Unpause so work can drain
            while (
                lane.status != LaneStatus.IDLE
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.05)

    def stats(self) -> dict:
        """Get statistics for all lanes."""
        return {
            "max_lanes": self._max_lanes,
            "active_lanes": self.active_lanes,
            "total_entries": self._next_id,
            "shutdown": self._shutdown,
            "lanes": {
                lid: {
                    "status": lane.status.value,
                    "queued": lane.queued_count,
                    "completed": lane.stats().completed,
                    "failed": lane.stats().failed,
                }
                for lid, lane in self._lanes.items()
            },
        }
