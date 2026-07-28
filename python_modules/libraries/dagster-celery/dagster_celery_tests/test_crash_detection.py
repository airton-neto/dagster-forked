import weakref
from unittest.mock import MagicMock, patch

import pytest
from dagster import DagsterRunStatus
from dagster._core.launcher import WorkerStatus
from dagster._core.launcher.base import ResumeRunContext
from dagster_celery.launcher import CeleryRunLauncher
from dagster_celery.tags import DAGSTER_CELERY_TASK_ID_TAG


@pytest.fixture
def mock_celery_app():
    app = MagicMock()
    app.AsyncResult.return_value = MagicMock()
    return app


@pytest.fixture
def launcher(mock_celery_app):
    """Create a CeleryRunLauncher with a mocked Celery app."""
    with patch.object(CeleryRunLauncher, "__init__", lambda self: None):
        obj = CeleryRunLauncher.__new__(CeleryRunLauncher)
        obj.celery = mock_celery_app
        # _instance is a read-only property backed by a weakref on MayHaveInstanceWeakref
        mock_instance = MagicMock()
        obj._instance_weakref = weakref.ref(mock_instance)  # noqa: SLF001
        obj._mock_instance_ref = mock_instance  # noqa: SLF001  # keep strong ref alive
        obj.default_queue = "dagster"
        obj.worker_health_confirmation_cycles = 3
        obj._worker_health_strikes = {}  # noqa: SLF001
        return obj


def _make_run(status=DagsterRunStatus.STARTED, task_id="test-task-123"):
    run = MagicMock()
    run.run_id = "test-run-id"
    run.status = status
    run.tags = {DAGSTER_CELERY_TASK_ID_TAG: task_id}
    run.job_code_origin = MagicMock()
    return run


class TestResumeRun:
    def test_resume_run_calls_create_resume_job_task_with_celery_app(self, launcher):
        """resume_run must pass self.celery (the Celery app), not args, to create_resume_job_task."""
        run = _make_run()
        context = ResumeRunContext(dagster_run=run, workspace=None, resume_attempt_number=1)

        with (
            patch("dagster_celery.launcher.create_resume_job_task") as mock_create,
            # Patch ResumeRunArgs so job_code_origin doesn't need to be a real JobPythonOrigin
            patch("dagster_celery.launcher.ResumeRunArgs"),
            # Patch pack_value and _launch_celery_task_run to isolate the unit under test
            patch("dagster_celery.launcher.pack_value"),
            patch.object(CeleryRunLauncher, "_launch_celery_task_run"),
        ):
            mock_task = MagicMock()
            mock_task.si.return_value = MagicMock()
            mock_task.si.return_value.apply_async.return_value = MagicMock(task_id="new-task-id")
            mock_create.return_value = mock_task

            launcher.resume_run(context)

            # The first positional arg must be the Celery app, not ResumeRunArgs
            mock_create.assert_called_once_with(launcher.celery)

            # The task signature must use resume_job_args_packed (not execute_job_args_packed)
            call_kwargs = mock_task.si.call_args.kwargs
            assert "resume_job_args_packed" in call_kwargs, (
                f"Expected 'resume_job_args_packed' kwarg, got: {list(call_kwargs.keys())}"
            )


class TestResumeRunRevokesPriorTask:
    """Resuming a run launches a second celery task for the same run id. The prior
    worker may still be alive (ping health checks can false-positive during a broker
    or network brownout), so the prior task must be revoked before the resume task is
    submitted — otherwise two workers execute the same run concurrently.
    """

    def test_resume_run_revokes_before_launching(self, launcher, mock_celery_app):
        run = _make_run(task_id="prior-task-id")
        context = ResumeRunContext(dagster_run=run, workspace=None, resume_attempt_number=1)
        order_tracker = MagicMock()

        with (
            patch("dagster_celery.launcher.create_resume_job_task") as mock_create,
            patch("dagster_celery.launcher.ResumeRunArgs"),
            patch("dagster_celery.launcher.pack_value"),
            patch.object(CeleryRunLauncher, "_launch_celery_task_run") as mock_launch,
        ):
            mock_create.return_value = MagicMock()
            order_tracker.attach_mock(mock_celery_app.AsyncResult.return_value.revoke, "revoke")
            order_tracker.attach_mock(mock_launch, "launch")

            launcher.resume_run(context)

        mock_celery_app.AsyncResult.assert_called_once_with("prior-task-id")
        mock_celery_app.AsyncResult.return_value.revoke.assert_called_once_with(terminate=True)
        call_names = [name for name, _args, _kwargs in order_tracker.mock_calls]
        assert call_names.index("revoke") < call_names.index("launch")

    def test_resume_run_revoke_failure_does_not_block_resume(self, launcher, mock_celery_app):
        run = _make_run(task_id="prior-task-id")
        mock_celery_app.AsyncResult.return_value.revoke.side_effect = Exception("broker down")

        with (
            patch("dagster_celery.launcher.create_resume_job_task") as mock_create,
            patch("dagster_celery.launcher.ResumeRunArgs"),
            patch("dagster_celery.launcher.pack_value"),
            patch.object(CeleryRunLauncher, "_launch_celery_task_run") as mock_launch,
        ):
            mock_create.return_value = MagicMock()
            context = ResumeRunContext(dagster_run=run, workspace=None, resume_attempt_number=1)
            launcher.resume_run(context)

            mock_launch.assert_called_once()


class TestCheckRunWorkerHealth:
    def test_task_success_returns_success(self, launcher, mock_celery_app):
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "SUCCESS"

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.SUCCESS

    def test_task_failure_returns_failed(self, launcher, mock_celery_app):
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "FAILURE"

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.FAILED

    def test_task_pending_returns_running_unconfirmed(self, launcher, mock_celery_app):
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "PENDING"

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING
        assert "unconfirmed" in health.msg.lower()

    def test_task_started_worker_alive_returns_running(self, launcher, mock_celery_app):
        """When Celery says STARTED and the worker responds to ping, return RUNNING."""
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "STARTED"
        result.info = {"hostname": "celery@worker-pod-1"}

        # Worker responds to ping
        inspect_mock = MagicMock()
        inspect_mock.ping.return_value = {"celery@worker-pod-1": {"ok": "pong"}}
        mock_celery_app.control.inspect.return_value = inspect_mock

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING

    def test_task_started_no_ping_reply_returns_running_unconfirmed(
        self, launcher, mock_celery_app
    ):
        """An empty ping reply is indistinguishable from a broker/network disruption:
        a live worker mid-run must not be declared FAILED off a single lost ping
        (which strands a zombie worker that later overwrites FAILURE with SUCCESS).
        """
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "STARTED"
        result.info = {"hostname": "celery@worker-pod-1"}

        # Worker does NOT respond to ping
        inspect_mock = MagicMock()
        inspect_mock.ping.return_value = None
        mock_celery_app.control.inspect.return_value = inspect_mock

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING
        assert "did not reply to ping" in health.msg.lower()

    def test_task_started_no_worker_info_returns_running_unconfirmed(
        self, launcher, mock_celery_app
    ):
        """When worker hostname is unavailable, count a strike instead of failing."""
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "STARTED"
        result.info = None

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING
        assert "hostname" in health.msg.lower()

    def test_task_started_inspect_raises_returns_running_unconfirmed(
        self, launcher, mock_celery_app
    ):
        """When inspect API fails (broker issue), count a strike instead of failing."""
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "STARTED"
        result.info = {"hostname": "celery@worker-pod-1"}

        # Inspect raises an exception
        mock_celery_app.control.inspect.side_effect = Exception("Broker connection failed")

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING
        assert "broker connection failed" in health.msg.lower()


class TestWorkerHealthConfirmation:
    """Soft health outcomes (empty ping reply, PENDING, broker errors) must be
    confirmed over consecutive monitoring cycles before FAILED is reported.

    Stock dagster core treats any non-RUNNING/SUCCESS status as unhealthy and acts
    immediately, so the confirmation MUST live in the launcher: unconfirmed cycles
    report RUNNING, and only the strike threshold escalates to FAILED.
    """

    def _no_ping_reply(self, mock_celery_app):
        result = mock_celery_app.AsyncResult.return_value
        result.state = "STARTED"
        result.info = {"hostname": "celery@worker-pod-1"}
        inspect_mock = MagicMock()
        inspect_mock.ping.return_value = None
        mock_celery_app.control.inspect.return_value = inspect_mock

    def _healthy_ping(self, mock_celery_app):
        inspect_mock = MagicMock()
        inspect_mock.ping.return_value = {"celery@worker-pod-1": {"ok": "pong"}}
        mock_celery_app.control.inspect.return_value = inspect_mock

    def test_strikes_escalate_to_failed_at_threshold(self, launcher, mock_celery_app):
        run = _make_run()
        self._no_ping_reply(mock_celery_app)

        # fixture sets worker_health_confirmation_cycles = 3
        for strike in (1, 2):
            health = launcher.check_run_worker_health(run)
            assert health.status == WorkerStatus.RUNNING
            assert f"{strike}/3" in health.msg

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.FAILED
        assert "3 consecutive" in health.msg

    def test_strikes_reset_after_failed_is_reported(self, launcher, mock_celery_app):
        run = _make_run()
        self._no_ping_reply(mock_celery_app)

        for _ in range(3):
            health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.FAILED

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING

    def test_healthy_ping_resets_strikes(self, launcher, mock_celery_app):
        run = _make_run()

        self._no_ping_reply(mock_celery_app)
        launcher.check_run_worker_health(run)
        launcher.check_run_worker_health(run)

        self._healthy_ping(mock_celery_app)
        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING
        assert not health.msg

        # Streak restarted: two more soft cycles stay below the threshold of 3
        self._no_ping_reply(mock_celery_app)
        launcher.check_run_worker_health(run)
        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING

    def test_strikes_are_tracked_per_run(self, launcher, mock_celery_app):
        self._no_ping_reply(mock_celery_app)

        run_a = _make_run()
        run_a.run_id = "run-a"
        run_b = _make_run()
        run_b.run_id = "run-b"

        launcher.check_run_worker_health(run_a)
        launcher.check_run_worker_health(run_a)
        launcher.check_run_worker_health(run_b)

        # run_a is at 2 strikes, run_b at 1 — neither has reached 3
        assert launcher.check_run_worker_health(run_b).status == WorkerStatus.RUNNING
        assert launcher.check_run_worker_health(run_a).status == WorkerStatus.FAILED

    def test_confirmation_cycles_config_default(self):
        field = CeleryRunLauncher.config_type()["worker_health_confirmation_cycles"]
        assert field.default_value == 5


class TestTerminate:
    def test_terminate_revokes_task(self, launcher, mock_celery_app):
        run = _make_run(task_id="task-9")
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001

        assert launcher.terminate("test-run-id") is True
        mock_celery_app.AsyncResult.assert_called_once_with("task-9")
        mock_celery_app.AsyncResult.return_value.revoke.assert_called_once_with(terminate=True)

    def test_terminate_without_task_id_tag_returns_false(self, launcher, mock_celery_app):
        """A run that never got a celery task id (launch failed early) must return a
        clean False, not raise KeyError.
        """
        run = _make_run()
        run.tags = {}
        launcher._instance.get_run_by_id.return_value = run  # noqa: SLF001

        assert launcher.terminate("test-run-id") is False
        mock_celery_app.AsyncResult.return_value.revoke.assert_not_called()
