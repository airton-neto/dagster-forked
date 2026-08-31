"""Worker-side ``ensure_task_dead`` control command.

Why this lives in the worker and not in the launcher: ``revoke(terminate=True)``
sends SIGTERM and then calls ``Request._announce_revoked`` ->
``celery.worker.state.task_ready``, which pops the request from ``requests`` and
discards it from ``active_requests``/``reserved_requests`` immediately
(``celery/worker/state.py:118-129``, ``celery/worker/request.py:413-417``,
celery 5.6.2). Two consequences killed the earlier launcher-side design:

- The fleet inventory reports the task gone the moment SIGTERM is sent, whether
  or not the child actually died, so no amount of polling can observe a zombie.
- A second ``revoke`` resolves its target through ``_find_requests_by_id`` over
  that same popped ``requests`` table (``celery/worker/control.py:115-121,228``),
  so the SIGKILL would reach nothing.

The billiard ``ApplyResult`` in ``pool._cache`` outlives ``task_ready`` and still
references the ``Request`` through its ``accept_callback`` bound method, so the
worker itself can find the process and kill it.
"""

import signal
from unittest.mock import MagicMock

import pytest
from dagster_celery.control import (
    MAX_GRACE_SECONDS,
    TERMINATE_GRACE_SECONDS,
    _coerce_grace_seconds,
    _ensure_task_dead_now,
    ensure_task_dead,
)

TASK_ID = "task-9"
WORKER_PID = 4242


class FakeRequest:
    def __init__(self, task_id):
        self.id = task_id

    def on_accepted(self, pid, time_accepted):
        """Bound method — celery passes this as the pool's accept_callback."""


class FakeApplyResult:
    """Shape of ``billiard.pool.ApplyResult`` (billiard 4.2.4)."""

    def __init__(self, task_id, worker_pid=WORKER_PID, is_ready=False, correlation_id=None):
        self._request = FakeRequest(task_id)
        self._accept_callback = self._request.on_accepted
        self._worker_pid = worker_pid
        self._is_ready = is_ready
        self.correlation_id = correlation_id

    def ready(self):
        return self._is_ready


class FinishesBeforeTheKill(FakeApplyResult):
    """Ready flips between the ownership check and the kill."""

    def __init__(self, task_id, flip_after=1):
        super().__init__(task_id)
        self._reads = 0
        self._flip_after = flip_after

    def ready(self):
        self._reads += 1
        return self._reads > self._flip_after


def _state(cache):
    """A control-command ``state`` whose consumer exposes a prefork pool."""
    state = MagicMock()
    state.consumer.pool._pool._cache = cache  # noqa: SLF001
    return state


def _kills(state):
    return [call.args for call in state.consumer.pool.terminate_job.call_args_list]


class TestEnsureTaskDeadNow:
    def test_kills_the_worker_process_when_the_task_is_still_running(self):
        state = _state({7: FakeApplyResult(TASK_ID)})

        assert _ensure_task_dead_now(state, TASK_ID) is True

        assert _kills(state) == [(WORKER_PID, signal.SIGKILL)]

    def test_matches_on_correlation_id_too(self):
        """Celery sets ``correlation_id=task_id`` on the ApplyResult."""
        result = FakeApplyResult("some-other-id", correlation_id=TASK_ID)
        state = _state({7: result})

        assert _ensure_task_dead_now(state, TASK_ID) is True

        assert _kills(state) == [(WORKER_PID, signal.SIGKILL)]

    def test_no_kill_when_the_task_is_not_in_the_cache(self):
        """The child exited on SIGTERM — the normal, healthy path."""
        state = _state({7: FakeApplyResult("a-different-task")})

        assert _ensure_task_dead_now(state, TASK_ID) is False

        state.consumer.pool.terminate_job.assert_not_called()

    def test_no_kill_when_the_result_is_already_ready(self):
        state = _state({7: FakeApplyResult(TASK_ID, is_ready=True)})

        assert _ensure_task_dead_now(state, TASK_ID) is False

        state.consumer.pool.terminate_job.assert_not_called()

    def test_no_kill_when_the_worker_pid_is_unknown(self):
        """A task accepted but never assigned a pid cannot be killed by pid."""
        state = _state({7: FakeApplyResult(TASK_ID, worker_pid=None)})

        assert _ensure_task_dead_now(state, TASK_ID) is False

        state.consumer.pool.terminate_job.assert_not_called()

    def test_empty_cache_is_a_no_op(self):
        state = _state({})

        assert _ensure_task_dead_now(state, TASK_ID) is False

        state.consumer.pool.terminate_job.assert_not_called()

    def test_a_broken_cache_entry_does_not_stop_the_walk(self):
        """One unusable entry must not hide the task behind it."""
        broken = MagicMock()
        broken.ready.side_effect = Exception("entry is garbage")
        state = _state({1: broken, 2: FakeApplyResult(TASK_ID)})

        assert _ensure_task_dead_now(state, TASK_ID) is True

        assert _kills(state) == [(WORKER_PID, signal.SIGKILL)]

    def test_missing_pool_never_raises(self):
        """A solo/threads pool has no ``_pool._cache``."""
        state = MagicMock()
        state.consumer.pool._pool = None  # noqa: SLF001

        assert _ensure_task_dead_now(state, TASK_ID) is False

    def test_no_kill_when_the_job_finishes_just_before_the_signal(self):
        """The job completes between the walk and the signal.

        Without the re-check, ``max-tasks-per-child`` recycling means the same
        pid can already be running the NEXT task, which would die instead.
        """
        result = FinishesBeforeTheKill(TASK_ID)
        state = _state({7: result})

        assert _ensure_task_dead_now(state, TASK_ID) is False

        state.consumer.pool.terminate_job.assert_not_called()

    def test_terminate_job_failure_never_raises(self):
        state = _state({7: FakeApplyResult(TASK_ID)})
        state.consumer.pool.terminate_job.side_effect = OSError("no such process")

        assert _ensure_task_dead_now(state, TASK_ID) is False


class TestEnsureTaskDeadCommand:
    """The control command schedules the kill; it must never block the worker."""

    def test_schedules_the_check_after_the_grace_period(self):
        timers = []

        def timer_factory(delay, function):
            timer = MagicMock()
            timers.append((delay, function, timer))
            return timer

        state = _state({7: FakeApplyResult(TASK_ID)})

        reply = ensure_task_dead(state, TASK_ID, timer_factory=timer_factory)

        assert len(timers) == 1
        delay, function, timer = timers[0]
        assert delay == TERMINATE_GRACE_SECONDS
        # Scheduled, not yet fired.
        state.consumer.pool.terminate_job.assert_not_called()
        timer.start.assert_called_once_with()
        assert timer.daemon is True
        assert reply == {"ok": f"scheduled SIGKILL check for {TASK_ID} in 20.0s"}

        # Firing the timer performs the kill.
        function()
        assert _kills(state) == [(WORKER_PID, signal.SIGKILL)]

    def test_grace_seconds_argument_is_respected(self):
        timers = []

        def timer_factory(delay, function):
            timers.append(delay)
            return MagicMock()

        state = _state({7: FakeApplyResult(TASK_ID)})

        ensure_task_dead(state, TASK_ID, grace_seconds=3.5, timer_factory=timer_factory)

        assert timers == [3.5]

    def test_zero_grace_runs_the_check_inline(self):
        state = _state({7: FakeApplyResult(TASK_ID)})

        ensure_task_dead(state, TASK_ID, grace_seconds=0)

        assert _kills(state) == [(WORKER_PID, signal.SIGKILL)]

    def test_a_failure_while_scheduling_never_propagates(self):
        """A control handler that raises would take down the worker's consumer."""

        def timer_factory(delay, function):
            raise RuntimeError("cannot spawn threads")

        state = _state({7: FakeApplyResult(TASK_ID)})

        reply = ensure_task_dead(state, TASK_ID, timer_factory=timer_factory)

        assert reply == {"error": "cannot spawn threads"}

    def test_registered_as_a_celery_control_command(self):
        from celery.worker.control import Panel

        assert "ensure_task_dead" in Panel.data
        assert Panel.data["ensure_task_dead"] is ensure_task_dead

    def test_imported_by_tasks_so_every_worker_registers_it(self):
        """The fleet's worker entry imports dagster_celery.tasks, so the command
        must be registered as a side effect of that import.
        """
        import dagster_celery.tasks as tasks_module

        assert tasks_module.ensure_task_dead is ensure_task_dead


class TestOnlyTheOwnerSchedulesATimer:
    """The broadcast has no destination, so it reaches every worker.

    A backfill canceling 500 runs would otherwise park ~500 daemon threads in
    EVERY worker main process inside one grace window. Only the worker that
    actually holds the task may schedule anything.
    """

    def _timer_factory(self, timers):
        def factory(delay, function):
            timers.append((delay, function))
            return MagicMock()

        return factory

    def test_a_worker_that_does_not_hold_the_task_schedules_nothing(self):
        timers = []
        state = _state({7: FakeApplyResult("someone-elses-task")})

        reply = ensure_task_dead(state, TASK_ID, timer_factory=self._timer_factory(timers))

        assert timers == []
        assert reply == {"ok": f"task {TASK_ID} not held by this worker"}

    def test_the_owning_worker_schedules_a_timer(self):
        timers = []
        state = _state({7: FakeApplyResult(TASK_ID)})

        reply = ensure_task_dead(state, TASK_ID, timer_factory=self._timer_factory(timers))

        assert len(timers) == 1
        assert reply == {"ok": f"scheduled SIGKILL check for {TASK_ID} in 20.0s"}

    def test_a_finished_task_is_not_an_owner(self):
        timers = []
        state = _state({7: FakeApplyResult(TASK_ID, is_ready=True)})

        ensure_task_dead(state, TASK_ID, timer_factory=self._timer_factory(timers))

        assert timers == []

    def test_a_worker_with_no_prefork_pool_schedules_nothing(self):
        timers = []
        state = MagicMock()
        state.consumer.pool._pool = None  # noqa: SLF001

        ensure_task_dead(state, TASK_ID, timer_factory=self._timer_factory(timers))

        assert timers == []

    def test_the_owner_re_walks_at_fire_time(self):
        """The task can still exit on its own inside the grace window."""
        timers = []
        result = FakeApplyResult(TASK_ID)
        state = _state({7: result})

        ensure_task_dead(state, TASK_ID, timer_factory=self._timer_factory(timers))
        _delay, fire = timers[0]

        # It finished during the window.
        result._is_ready = True  # noqa: SLF001
        fire()

        state.consumer.pool.terminate_job.assert_not_called()


class TestGraceSecondsCoercion:
    """``@control_command(args=...)`` declares types but validates nothing."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (5, 5.0),
            ("7.5", 7.5),
            (None, TERMINATE_GRACE_SECONDS),
            ("not-a-number", TERMINATE_GRACE_SECONDS),
            ([], TERMINATE_GRACE_SECONDS),
            (float("nan"), TERMINATE_GRACE_SECONDS),
            (-4.0, 0.0),
            (99999.0, MAX_GRACE_SECONDS),
        ],
    )
    def test_coercion(self, raw, expected):
        assert _coerce_grace_seconds(raw) == expected

    def test_a_garbage_grace_value_does_not_raise(self):
        timers = []

        def factory(delay, function):
            timers.append(delay)
            return MagicMock()

        state = _state({7: FakeApplyResult(TASK_ID)})

        ensure_task_dead(state, TASK_ID, grace_seconds="junk", timer_factory=factory)

        assert timers == [TERMINATE_GRACE_SECONDS]

    def test_an_absurd_grace_value_is_clamped(self):
        timers = []

        def factory(delay, function):
            timers.append(delay)
            return MagicMock()

        state = _state({7: FakeApplyResult(TASK_ID)})

        ensure_task_dead(state, TASK_ID, grace_seconds=86400, timer_factory=factory)

        assert timers == [MAX_GRACE_SECONDS]


@pytest.mark.parametrize("grace", [0, 0.0])
def test_falsy_grace_still_checks(grace):
    state = _state({7: FakeApplyResult(TASK_ID)})

    ensure_task_dead(state, TASK_ID, grace_seconds=grace)

    assert _kills(state) == [(WORKER_PID, signal.SIGKILL)]
