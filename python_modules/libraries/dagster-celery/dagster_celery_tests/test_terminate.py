"""Termination path of ``CeleryRunLauncher``.

Incident 2026-08-31 (auren-aes): run monitoring hit ``dagster/max_runtime`` on
``Tucano_Intraday_Job``, ``terminate()`` sent SIGTERM through
``revoke(terminate=True)`` and the run was marked FAILED — but the billiard
child never died. Dagster's op-concurrency-pool wait loop does not check the
captured interrupt, so the SIGTERM only surfaced 14 hours later when the step
finally started. Each zombie held one Celery slot (``concurrency=1`` workers)
until six of them wedged the whole intraday fleet.

``terminate()`` must stay non-blocking: it is called from the webserver request
thread (``queued_run_coordinator.py:341``), from the backfill daemon in a serial
loop over up to 500 runs (``backfill.py:606``), and from run monitoring. The
confirmation therefore happens on the worker, through the ``ensure_task_dead``
control command — see ``dagster_celery/control.py``.
"""

import weakref
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
from dagster import DagsterRunStatus
from dagster_celery.control import TERMINATE_GRACE_SECONDS
from dagster_celery.launcher import CeleryRunLauncher
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
        obj._worker_health_strikes = {}  # noqa: SLF001
        obj._control_app = lambda: nullcontext(mock_celery_app)  # noqa: SLF001
        return obj


def _make_run(status=DagsterRunStatus.CANCELING, task_id=TASK_ID):
    run = MagicMock()
    run.run_id = "test-run-id"
    run.status = status
    run.tags = {DAGSTER_CELERY_TASK_ID_TAG: task_id}
    return run


def _revoke_calls(mock_celery_app):
    return mock_celery_app.AsyncResult.return_value.revoke.call_args_list


class TestTerminate:
    def test_sends_sigterm_and_broadcasts_the_kill_check(self, launcher, mock_celery_app):
        run = _make_run()
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001

        assert launcher.terminate("test-run-id") is True

        mock_celery_app.AsyncResult.assert_called_once_with(TASK_ID)
        # Exactly one revoke: the SIGTERM. There is no second, SIGKILL revoke —
        # celery would resolve it against a requests table the first revoke
        # already emptied (celery/worker/control.py:115-121,228).
        assert _revoke_calls(mock_celery_app) == [((), {"terminate": True})]

        mock_celery_app.control.broadcast.assert_called_once_with(
            "ensure_task_dead",
            arguments={
                "task_id": TASK_ID,
                "grace_seconds": TERMINATE_GRACE_SECONDS,
            },
            reply=False,
        )

    def test_revoke_happens_before_the_broadcast(self, launcher, mock_celery_app):
        """The grace window starts at SIGTERM, so SIGTERM must go first."""
        run = _make_run()
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001

        order = MagicMock()
        order.attach_mock(mock_celery_app.AsyncResult.return_value.revoke, "revoke")
        order.attach_mock(mock_celery_app.control.broadcast, "broadcast")

        launcher.terminate("test-run-id")

        names = [call[0] for call in order.mock_calls]
        assert names.index("revoke") < names.index("broadcast")

    def test_does_not_block(self, launcher, mock_celery_app):
        """No sleeping and no reply-waiting: terminate() runs on the webserver
        request thread and in a 500-run backfill loop.
        """
        run = _make_run()
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001

        with patch("dagster_celery.launcher.time.sleep") as mock_sleep:
            launcher.terminate("test-run-id")

        mock_sleep.assert_not_called()
        mock_celery_app.control.inspect.assert_not_called()
        assert mock_celery_app.control.broadcast.call_args.kwargs["reply"] is False

    def test_broadcast_failure_does_not_fail_terminate(self, launcher, mock_celery_app):
        """A dead broker must not turn a terminate into an exception — the
        SIGTERM is already out.
        """
        run = _make_run()
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001
        mock_celery_app.control.broadcast.side_effect = Exception("broker down")

        assert launcher.terminate("test-run-id") is True

        assert _revoke_calls(mock_celery_app) == [((), {"terminate": True})]

    def test_broadcast_goes_through_the_short_lived_control_app(self, launcher, mock_celery_app):
        """Control-plane calls must not use the long-lived cached app, whose
        reply routing can rot in the daemon process.
        """
        run = _make_run()
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001
        sentinel_app = MagicMock()
        launcher._control_app = lambda: nullcontext(sentinel_app)  # noqa: SLF001

        launcher.terminate("test-run-id")

        sentinel_app.control.broadcast.assert_called_once()
        mock_celery_app.control.broadcast.assert_not_called()

    def test_canceling_run_is_not_reported_canceled(self, launcher, mock_celery_app):
        """The launcher must NOT resolve the CANCELING run itself.

        ``check_run_timeout`` reports CANCELING, calls terminate(), then
        force-marks the run FAILED; ``_force_mark_as_failed`` skips a run that is
        already finished, so a report_run_canceled() here would flip the
        max-runtime outcome from FAILURE to CANCELED.
        """
        run = _make_run(status=DagsterRunStatus.CANCELING)
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001

        assert launcher.terminate("test-run-id") is True

        launcher._instance.report_run_canceled.assert_not_called()  # noqa: SLF001
        launcher._instance.report_run_failed.assert_not_called()  # noqa: SLF001

    def test_missing_task_id_tag_returns_false_without_revoking(self, launcher, mock_celery_app):
        run = _make_run()
        run.tags = {}
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001

        assert launcher.terminate("test-run-id") is False

        mock_celery_app.AsyncResult.return_value.revoke.assert_not_called()
        mock_celery_app.control.broadcast.assert_not_called()

    def test_unknown_run_returns_false_without_revoking(self, launcher, mock_celery_app):
        launcher._instance.get_run_by_id.return_value = None  # noqa: SLF001

        assert launcher.terminate("test-run-id") is False

        mock_celery_app.AsyncResult.return_value.revoke.assert_not_called()
        mock_celery_app.control.broadcast.assert_not_called()


class TestResumeRunStaysSigtermOnly:
    def test_resume_run_prior_revoke_does_not_broadcast(self, launcher, mock_celery_app):
        """resume_run's prior-task revoke targets a possibly healthy, partitioned
        worker — it must stay a plain SIGTERM with no kill escalation.
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
        mock_celery_app.control.broadcast.assert_not_called()


class TestTerminateGraceIsInternal:
    """The grace window is a correctness guarantee of terminate(), not an
    operator knob. It must stay off the config schema.
    """

    def test_grace_window_is_twenty_seconds(self):
        assert TERMINATE_GRACE_SECONDS == 20.0

    def test_not_exposed_as_a_config_field(self):
        assert "terminate_grace_seconds" not in CeleryRunLauncher.config_type()

    def test_the_removed_config_key_is_rejected_by_the_real_config_path(self):
        """A dagster.yaml still carrying the old field must fail loudly.

        This is the production path: dagster validates the launcher config with
        ``process_config`` against ``config_type()`` before ``from_config_value``
        is ever reached, so the stale key is caught there.
        """
        from dagster._config import process_config

        result = process_config(
            CeleryRunLauncher.config_type(),
            {"default_queue": "dagster", "terminate_grace_seconds": 45.0},
        )

        assert not result.success
        assert any("terminate_grace_seconds" in str(error.message) for error in result.errors or [])

    def test_a_valid_config_still_processes(self):
        """Guard against the previous test passing for the wrong reason."""
        from dagster._config import process_config

        result = process_config(CeleryRunLauncher.config_type(), {"default_queue": "dagster"})

        assert result.success
