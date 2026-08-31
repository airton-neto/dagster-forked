"""Termination path of ``CeleryRunLauncher``.

Incident 2026-08-31 (auren-aes): run monitoring hit ``dagster/max_runtime`` on
``Tucano_Intraday_Job``, ``terminate()`` sent SIGTERM through
``revoke(terminate=True)`` and the run was marked FAILED — but the billiard
child never died. Dagster's op-concurrency-pool wait loop does not check the
captured interrupt, so the SIGTERM only surfaced 14 hours later when the step
finally started. Each zombie held one Celery slot (``concurrency=1`` workers)
until six of them wedged the whole intraday fleet.
"""

import weakref
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
from dagster import DagsterRunStatus
from dagster_celery.launcher import (
    TERMINATE_GRACE_SECONDS,
    TERMINATE_POLL_INTERVAL_SECONDS,
    CeleryRunLauncher,
)
from dagster_celery.tags import DAGSTER_CELERY_TASK_ID_TAG

TASK_ID = "task-9"


@pytest.fixture
def mock_celery_app():
    app = MagicMock()
    app.AsyncResult.return_value = MagicMock()
    return app


@pytest.fixture
def launcher(mock_celery_app):
    """A ``CeleryRunLauncher`` with a mocked Celery app (same seam as test_crash_detection)."""
    with patch.object(CeleryRunLauncher, "__init__", lambda self: None):
        obj = CeleryRunLauncher.__new__(CeleryRunLauncher)
        obj.celery = mock_celery_app
        mock_instance = MagicMock()
        obj._instance_weakref = weakref.ref(mock_instance)  # noqa: SLF001
        obj._mock_instance_ref = mock_instance  # noqa: SLF001  # keep strong ref alive
        obj.default_queue = "dagster"
        obj.worker_health_confirmation_cycles = 3
        obj.ping_timeout = 10.0
        obj.terminate_grace_seconds = TERMINATE_GRACE_SECONDS
        obj._worker_health_strikes = {}  # noqa: SLF001
        obj._control_app = lambda: nullcontext(mock_celery_app)  # noqa: SLF001
        return obj


class FakeClock:
    """Monotonic clock that only advances when the code under test sleeps."""

    def __init__(self, now=1000.0):
        self.now = now
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock():
    fake = FakeClock()
    with (
        patch("dagster_celery.launcher.time.monotonic", fake.monotonic),
        patch("dagster_celery.launcher.time.sleep", fake.sleep),
    ):
        yield fake


def _make_run(status=DagsterRunStatus.CANCELING, task_id=TASK_ID):
    run = MagicMock()
    run.run_id = "test-run-id"
    run.status = status
    run.tags = {DAGSTER_CELERY_TASK_ID_TAG: task_id}
    return run


def _holding(task_id=TASK_ID):
    """One inventory reply where a worker still holds the task."""
    return {"celery@worker-intraday-0": [{"id": task_id, "args": "[]", "kwargs": "{}"}]}


def _answered_empty():
    """Workers replied and none of them holds any task — evidence the task is gone."""
    return {"celery@worker-intraday-0": [], "celery@worker-intraday-1": []}


def _silence():
    """Nobody replied at all — inconclusive, not evidence of anything."""
    return {}


def _inventory(mock_celery_app, active_replies, reserved_replies=None):
    inspector = mock_celery_app.control.inspect.return_value
    inspector.active.side_effect = list(active_replies)
    inspector.reserved.side_effect = (
        list(reserved_replies)
        if reserved_replies is not None
        else [_answered_empty() for _ in active_replies]
    )


def _revoke_calls(mock_celery_app):
    return mock_celery_app.AsyncResult.return_value.revoke.call_args_list


def _event_metadata(launcher):
    """Unwrap the EngineEventData metadata of the single reported engine event."""
    event_data = launcher._instance.report_engine_event.call_args.args[2]  # noqa: SLF001
    return {key: value.value for key, value in event_data.metadata.items()}


class TestTerminateEscalation:
    def test_sigterm_only_when_task_leaves_the_fleet_within_the_grace_window(
        self, launcher, mock_celery_app, clock
    ):
        """The task dies on SIGTERM: exactly one revoke, and no SIGKILL."""
        run = _make_run()
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001
        _inventory(mock_celery_app, [_holding(), _holding(), _answered_empty()])

        assert launcher.terminate("test-run-id") is True

        mock_celery_app.AsyncResult.assert_called_once_with(TASK_ID)
        assert _revoke_calls(mock_celery_app) == [((), {"terminate": True})]
        assert mock_celery_app.control.inspect.call_count == 3
        assert clock.sleeps == [TERMINATE_POLL_INTERVAL_SECONDS, TERMINATE_POLL_INTERVAL_SECONDS]

    def test_sigkill_when_the_task_is_still_active_at_the_deadline(
        self, launcher, mock_celery_app, clock
    ):
        """The zombie case: SIGTERM ignored for the whole window, so escalate."""
        run = _make_run()
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001
        _inventory(mock_celery_app, [_holding() for _ in range(20)])

        assert launcher.terminate("test-run-id") is True

        assert _revoke_calls(mock_celery_app) == [
            ((), {"terminate": True}),
            ((), {"terminate": True, "signal": "SIGKILL"}),
        ]
        # 20s grace / 2s poll interval
        assert clock.sleeps == [TERMINATE_POLL_INTERVAL_SECONDS] * 10

        launcher._instance.report_engine_event.assert_called_once()  # noqa: SLF001
        message = launcher._instance.report_engine_event.call_args.args[0]  # noqa: SLF001
        assert message == (
            f"Celery task {TASK_ID} was still held by a worker 20.0s after SIGTERM;"
            " escalating the revoke to SIGKILL."
        )
        assert launcher._instance.report_engine_event.call_args.args[1] is run  # noqa: SLF001
        event_metadata = _event_metadata(launcher)
        assert event_metadata["Run ID"] == "test-run-id"
        assert event_metadata["Celery Task ID"] == TASK_ID
        assert event_metadata["Terminate Grace Seconds"] == 20.0
        assert event_metadata["Fleet Inventory"] == "task still held by a worker"

    def test_poll_timeout_is_short_enough_to_stay_inside_the_grace_window(
        self, launcher, mock_celery_app, clock
    ):
        """A per-poll inspect timeout of ping_timeout (10s) would blow a 20s window."""
        run = _make_run()
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001
        _inventory(mock_celery_app, [_holding() for _ in range(20)])

        launcher.terminate("test-run-id")

        assert (
            mock_celery_app.control.inspect.call_args_list
            == [((), {"timeout": TERMINATE_POLL_INTERVAL_SECONDS})] * 10
        )

    def test_grace_seconds_zero_sends_a_single_sigterm_and_never_polls(
        self, launcher, mock_celery_app, clock
    ):
        run = _make_run()
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001
        launcher.terminate_grace_seconds = 0.0

        assert launcher.terminate("test-run-id") is True

        assert _revoke_calls(mock_celery_app) == [((), {"terminate": True})]
        mock_celery_app.control.inspect.assert_not_called()
        assert clock.sleeps == []

    def test_inconclusive_inventory_for_the_whole_window_escalates_to_sigkill(
        self, launcher, mock_celery_app, clock
    ):
        """Nobody ever replies. The run is already being terminated, and a live
        zombie holding a concurrency=1 slot is the worse outcome than a redundant
        SIGKILL against a task that already exited.
        """
        run = _make_run()
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001
        _inventory(
            mock_celery_app,
            [_silence() for _ in range(20)],
            [_silence() for _ in range(20)],
        )

        assert launcher.terminate("test-run-id") is True

        assert _revoke_calls(mock_celery_app) == [
            ((), {"terminate": True}),
            ((), {"terminate": True, "signal": "SIGKILL"}),
        ]
        assert _event_metadata(launcher)["Fleet Inventory"] == "inconclusive (no worker replied)"
        assert launcher._instance.report_engine_event.call_args.args[0] == (  # noqa: SLF001
            f"Celery task {TASK_ID} could not be confirmed dead (no worker replied)"
            " 20.0s after SIGTERM; escalating the revoke to SIGKILL."
        )

    def test_inventory_failure_does_not_break_termination(self, launcher, mock_celery_app, clock):
        """A broker error while polling is inconclusive, not a reason to raise."""
        run = _make_run()
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001
        mock_celery_app.control.inspect.side_effect = Exception("broker down")

        assert launcher.terminate("test-run-id") is True

        assert _revoke_calls(mock_celery_app) == [
            ((), {"terminate": True}),
            ((), {"terminate": True, "signal": "SIGKILL"}),
        ]

    def test_canceling_run_is_not_reported_canceled_after_sigkill(
        self, launcher, mock_celery_app, clock
    ):
        """The launcher must NOT resolve the CANCELING run itself.

        ``check_run_timeout`` reports CANCELING, calls terminate(), then
        force-marks the run FAILED; ``_force_mark_as_failed`` skips a run that is
        already finished, so a report_run_canceled() here would flip the
        max-runtime outcome from FAILURE to CANCELED. A genuinely stuck CANCELING
        run is resolved by ``monitor_canceling_run`` instead.
        """
        run = _make_run(status=DagsterRunStatus.CANCELING)
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001
        _inventory(mock_celery_app, [_holding() for _ in range(20)])

        assert launcher.terminate("test-run-id") is True

        launcher._instance.report_run_canceled.assert_not_called()  # noqa: SLF001
        launcher._instance.report_run_failed.assert_not_called()  # noqa: SLF001

    def test_missing_task_id_tag_returns_false_without_revoking(
        self, launcher, mock_celery_app, clock
    ):
        run = _make_run()
        run.tags = {}
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001

        assert launcher.terminate("test-run-id") is False

        mock_celery_app.AsyncResult.return_value.revoke.assert_not_called()
        mock_celery_app.control.inspect.assert_not_called()

    def test_unknown_run_returns_false_without_revoking(self, launcher, mock_celery_app, clock):
        launcher._instance.get_run_by_id.return_value = None  # noqa: SLF001

        assert launcher.terminate("test-run-id") is False

        mock_celery_app.AsyncResult.return_value.revoke.assert_not_called()


class TestResumeRunStaysSigtermOnly:
    def test_resume_run_prior_revoke_never_escalates(self, launcher, mock_celery_app, clock):
        """resume_run's prior-task revoke targets a possibly healthy, partitioned
        worker — it must stay a plain SIGTERM.
        """
        run = _make_run(status=DagsterRunStatus.STARTED)
        context = MagicMock()
        context.dagster_run = run

        with (
            patch("dagster_celery.launcher.create_resume_job_task"),
            patch("dagster_celery.launcher.ResumeRunArgs"),
            patch("dagster_celery.launcher.pack_value"),
            patch.object(CeleryRunLauncher, "_launch_celery_task_run"),
        ):
            launcher.resume_run(context)

        assert _revoke_calls(mock_celery_app) == [((), {"terminate": True})]
        mock_celery_app.control.inspect.assert_not_called()


class TestTerminateGraceIsInternal:
    """The grace window is a correctness guarantee of terminate(), not an
    operator knob. It must stay off the config schema, and every launcher must
    get it.
    """

    def test_grace_window_is_twenty_seconds(self):
        assert TERMINATE_GRACE_SECONDS == 20.0

    def test_not_exposed_as_a_config_field(self):
        assert "terminate_grace_seconds" not in CeleryRunLauncher.config_type()

    def test_launcher_built_from_config_gets_the_constant(self):
        launcher = CeleryRunLauncher.from_config_value(
            None,  # pyright: ignore[reportArgumentType]
            {"default_queue": "dagster"},
        )
        assert launcher.terminate_grace_seconds == TERMINATE_GRACE_SECONDS

    def test_config_value_for_the_removed_field_is_rejected(self):
        """A dagster.yaml carrying the old field must fail loudly, not silently."""
        with pytest.raises(TypeError):
            CeleryRunLauncher.from_config_value(
                None,  # pyright: ignore[reportArgumentType]
                {"default_queue": "dagster", "terminate_grace_seconds": 45.0},
            )

    def test_instance_attribute_override_drives_the_deadline(
        self, launcher, mock_celery_app, clock
    ):
        """Tests lower the window; the loop must honour the instance attribute."""
        run = _make_run()
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001
        launcher.terminate_grace_seconds = 6.0
        _inventory(mock_celery_app, [_holding() for _ in range(10)])

        assert launcher.terminate("test-run-id") is True

        # 6s window / 2s poll interval
        assert clock.sleeps == [TERMINATE_POLL_INTERVAL_SECONDS] * 3
        assert _revoke_calls(mock_celery_app) == [
            ((), {"terminate": True}),
            ((), {"terminate": True, "signal": "SIGKILL"}),
        ]
        assert _event_metadata(launcher)["Terminate Grace Seconds"] == 6.0
