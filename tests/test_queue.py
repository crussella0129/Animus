"""Tests for lane-based command queueing."""

import asyncio

import pytest
from src.core.queue import (
    CommandQueue,
    Lane,
    LaneStatus,
    QueueEntry,
    LaneStats,
)


class TestQueueEntry:
    """Tests for QueueEntry."""

    def test_priority_ordering(self):
        a = QueueEntry(id=1, lane_id="x", coroutine=asyncio.sleep(0), priority=5)
        b = QueueEntry(id=2, lane_id="x", coroutine=asyncio.sleep(0), priority=10)
        # Higher priority first
        assert b < a

    def test_fifo_ordering_same_priority(self):
        a = QueueEntry(id=1, lane_id="x", coroutine=asyncio.sleep(0), priority=0)
        b = QueueEntry(id=2, lane_id="x", coroutine=asyncio.sleep(0), priority=0)
        # Lower id first (FIFO)
        assert a < b


class TestLane:
    """Tests for Lane."""

    @pytest.mark.asyncio
    async def test_initial_state(self):
        lane = Lane("test")
        assert lane.lane_id == "test"
        assert lane.status == LaneStatus.IDLE
        assert lane.queued_count == 0
        assert not lane.is_paused

    @pytest.mark.asyncio
    async def test_enqueue_and_process(self):
        lane = Lane("test")
        results = []

        async def work(value):
            results.append(value)
            return value

        entry = QueueEntry(
            id=1, lane_id="test",
            coroutine=work(42),
        )
        await lane.enqueue(entry)

        # Wait for processing
        await asyncio.sleep(0.1)
        assert 42 in results

    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        lane = Lane("test")
        order = []

        async def work(value):
            order.append(value)
            await asyncio.sleep(0.01)
            return value

        for i in range(3):
            entry = QueueEntry(id=i, lane_id="test", coroutine=work(i))
            await lane.enqueue(entry)

        await asyncio.sleep(0.2)
        assert order == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_pause_resume(self):
        lane = Lane("test")
        lane.pause()
        assert lane.is_paused
        assert lane.status == LaneStatus.IDLE  # wasn't running

        lane.resume()
        assert not lane.is_paused

    @pytest.mark.asyncio
    async def test_stats(self):
        lane = Lane("test")
        stats = lane.stats()
        assert stats.lane_id == "test"
        assert stats.status == LaneStatus.IDLE
        assert stats.queued == 0
        assert stats.completed == 0
        assert stats.failed == 0

    @pytest.mark.asyncio
    async def test_failed_entry(self):
        lane = Lane("test")

        async def fail():
            raise ValueError("boom")

        entry = QueueEntry(id=1, lane_id="test", coroutine=fail())
        await lane.enqueue(entry)

        await asyncio.sleep(0.1)
        stats = lane.stats()
        assert stats.failed == 1


class TestCommandQueue:
    """Tests for CommandQueue."""

    @pytest.mark.asyncio
    async def test_basic_submit(self):
        queue = CommandQueue()

        async def work():
            return 42

        entry = await queue.submit("lane-1", work())
        assert entry.result == 42
        assert entry.error is None
        assert entry.completed_at is not None

    @pytest.mark.asyncio
    async def test_sequential_within_lane(self):
        queue = CommandQueue()
        order = []

        async def work(value):
            order.append(value)
            await asyncio.sleep(0.01)
            return value

        # Submit 3 items to same lane
        entries = await asyncio.gather(
            queue.submit("lane-1", work(1)),
            queue.submit("lane-1", work(2)),
            queue.submit("lane-1", work(3)),
        )

        assert order == [1, 2, 3]
        assert all(e.result is not None for e in entries)

    @pytest.mark.asyncio
    async def test_parallel_across_lanes(self):
        queue = CommandQueue(max_lanes=4)
        started = []

        async def work(lane):
            started.append(lane)
            await asyncio.sleep(0.05)
            return lane

        # Submit to different lanes — should run in parallel
        entries = await asyncio.gather(
            queue.submit("a", work("a")),
            queue.submit("b", work("b")),
        )

        assert set(e.result for e in entries) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        queue = CommandQueue()
        order = []

        async def work(value):
            order.append(value)
            return value

        # Submit low priority first, then high priority
        # Due to async scheduling, we need to pause the lane first
        lane = queue._get_or_create_lane("test")
        lane.pause()

        entry_low = QueueEntry(id=1, lane_id="test", coroutine=work("low"), priority=1)
        entry_high = QueueEntry(id=2, lane_id="test", coroutine=work("high"), priority=10)

        await lane.enqueue(entry_low)
        await lane.enqueue(entry_high)
        lane.resume()

        await asyncio.sleep(0.1)
        # High priority should execute first
        assert order[0] == "high"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        queue = CommandQueue()

        async def fail():
            raise ValueError("test error")

        entry = await queue.submit("lane-1", fail())
        assert entry.error is not None
        assert "test error" in str(entry.error)

    @pytest.mark.asyncio
    async def test_pause_lane(self):
        queue = CommandQueue()

        async def work():
            return 1

        await queue.submit("test", work())
        assert queue.pause_lane("test")
        assert queue.get_lane_status("test") in (LaneStatus.PAUSED, LaneStatus.IDLE)

    @pytest.mark.asyncio
    async def test_resume_lane(self):
        queue = CommandQueue()

        async def work():
            return 1

        await queue.submit("test", work())
        queue.pause_lane("test")
        assert queue.resume_lane("test")

    def test_pause_nonexistent_lane(self):
        queue = CommandQueue()
        assert not queue.pause_lane("nonexistent")

    def test_resume_nonexistent_lane(self):
        queue = CommandQueue()
        assert not queue.resume_lane("nonexistent")

    @pytest.mark.asyncio
    async def test_max_lanes_enforcement(self):
        queue = CommandQueue(max_lanes=2)

        async def long_work():
            await asyncio.sleep(1.0)
            return True

        # Fill up lanes with long-running work
        asyncio.create_task(queue.submit("a", long_work()))
        asyncio.create_task(queue.submit("b", long_work()))
        await asyncio.sleep(0.05)  # Let them start

        # Third lane should raise
        with pytest.raises(RuntimeError, match="Maximum lanes"):
            queue._get_or_create_lane("c")

    @pytest.mark.asyncio
    async def test_shutdown(self):
        queue = CommandQueue()

        async def work():
            return 1

        await queue.submit("test", work())
        await queue.shutdown(timeout=1.0)

        with pytest.raises(RuntimeError, match="shut down"):
            await queue.submit("test", work())

    @pytest.mark.asyncio
    async def test_stats(self):
        queue = CommandQueue(max_lanes=4)

        async def work():
            return 1

        await queue.submit("lane-1", work())
        stats = queue.stats()
        assert stats["max_lanes"] == 4
        assert stats["total_entries"] == 1
        assert "lane-1" in stats["lanes"]

    def test_get_lane_status_nonexistent(self):
        queue = CommandQueue()
        assert queue.get_lane_status("nope") is None

    @pytest.mark.asyncio
    async def test_active_lanes_count(self):
        queue = CommandQueue()

        async def work():
            return 1

        assert queue.active_lanes == 0
        await queue.submit("a", work())
        # After completion, lane should be idle
        await asyncio.sleep(0.05)
        assert queue.active_lanes == 0  # Completed

    @pytest.mark.asyncio
    async def test_lane_reuse_when_idle(self):
        queue = CommandQueue(max_lanes=1)

        async def work():
            return 1

        # First lane completes
        await queue.submit("first", work())
        await asyncio.sleep(0.05)

        # Should reuse the lane slot
        await queue.submit("second", work())
