import weakref
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from celery import Celery
from celery.exceptions import Ignore
from dagster import DagsterRunStatus
from dagster._core.launcher import WorkerStatus
from dagster._core.launcher.base import ResumeRunContext
from dagster_celery.launcher import (
    HEALTH_CHECK_MAX_RETRIES,
    TASK_SUCCESS_TERMINAL_GRACE_SECONDS,
    CeleryRunLauncher,
)
from dagster_celery.tags import DAGSTER_CELERY_TASK_ID_TAG, DAGSTER_CELERY_WORKER_HOSTNAME_TAG
from dagster_celery.tasks import create_execute_job_task


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

    def _resume(self, launcher, run):
        context = ResumeRunContext(dagster_run=run, workspace=None, resume_attempt_number=1)
        with (
            patch("dagster_celery.launcher.create_resume_job_task") as mock_create,
            patch("dagster_celery.launcher.ResumeRunArgs"),
            patch("dagster_celery.launcher.pack_value"),
            patch.object(CeleryRunLauncher, "_launch_celery_task_run"),
        ):
            mock_task = MagicMock()
            mock_create.return_value = mock_task
            launcher.resume_run(context)

    def test_resume_run_revokes_prior_task(self, launcher, mock_celery_app):
        run = _make_run(task_id="prior-task-id")

        self._resume(launcher, run)

        mock_celery_app.AsyncResult.assert_called_once_with("prior-task-id")
        mock_celery_app.AsyncResult.return_value.revoke.assert_called_once_with(terminate=True)

    def test_resume_run_without_prior_task_tag_does_not_revoke(self, launcher, mock_celery_app):
        run = _make_run()
        run.tags = {}

        self._resume(launcher, run)

        mock_celery_app.AsyncResult.return_value.revoke.assert_not_called()

    def test_resume_run_revoke_failure_does_not_block_resume(self, launcher, mock_celery_app):
        """A broker error while revoking must not prevent the resume from launching."""
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


def _make_current_run(finished: bool, status=DagsterRunStatus.STARTED):
    current = MagicMock()
    current.run_id = "test-run-id"
    current.is_finished = finished
    current.status = status
    return current


class TestCheckRunWorkerHealth:
    def test_task_success_run_finished_returns_success(self, launcher, mock_celery_app):
        run = _make_run()
        launcher._instance.get_run_by_id.return_value = _make_current_run(finished=True)  # noqa: SLF001
        result = mock_celery_app.AsyncResult.return_value
        result.state = "SUCCESS"

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.SUCCESS

    def test_task_success_run_not_finished_past_grace_returns_failed(
        self, launcher, mock_celery_app
    ):
        """Task SUCCESS but run never reached a terminal state (e.g. the result was
        overwritten by a duplicate delivery no-op) must be reported unhealthy.
        """
        run = _make_run()
        launcher._instance.get_run_by_id.return_value = _make_current_run(finished=False)  # noqa: SLF001
        result = mock_celery_app.AsyncResult.return_value
        result.state = "SUCCESS"
        result.date_done = datetime.now(timezone.utc) - timedelta(
            seconds=TASK_SUCCESS_TERMINAL_GRACE_SECONDS + 60
        )

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.FAILED
        assert "terminal" in health.msg.lower()

    def test_task_success_run_not_finished_within_grace_returns_running(
        self, launcher, mock_celery_app
    ):
        """Right after task completion the run terminal event may still be in flight."""
        run = _make_run()
        launcher._instance.get_run_by_id.return_value = _make_current_run(finished=False)  # noqa: SLF001
        result = mock_celery_app.AsyncResult.return_value
        result.state = "SUCCESS"
        result.date_done = datetime.now(timezone.utc)

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING

    def test_task_success_run_not_finished_no_date_done_returns_failed(
        self, launcher, mock_celery_app
    ):
        run = _make_run()
        launcher._instance.get_run_by_id.return_value = _make_current_run(finished=False)  # noqa: SLF001
        result = mock_celery_app.AsyncResult.return_value
        result.state = "SUCCESS"
        result.date_done = None

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.FAILED

    def test_task_success_naive_date_done_is_treated_as_utc(self, launcher, mock_celery_app):
        run = _make_run()
        launcher._instance.get_run_by_id.return_value = _make_current_run(finished=False)  # noqa: SLF001
        result = mock_celery_app.AsyncResult.return_value
        result.state = "SUCCESS"
        result.date_done = datetime.utcnow() - timedelta(
            seconds=TASK_SUCCESS_TERMINAL_GRACE_SECONDS + 60
        )

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.FAILED

    def test_pending_with_hostname_tag_pings_worker_and_returns_running(
        self, launcher, mock_celery_app
    ):
        """PENDING may mean the result backend lost the task meta (e.g. redis restart).
        If the run is tagged with the worker hostname, ping it instead of giving up.
        """
        run = _make_run()
        run.tags[DAGSTER_CELERY_WORKER_HOSTNAME_TAG] = "celery@worker-pod-1"
        result = mock_celery_app.AsyncResult.return_value
        result.state = "PENDING"

        inspect_mock = MagicMock()
        inspect_mock.ping.return_value = {"celery@worker-pod-1": {"ok": "pong"}}
        mock_celery_app.control.inspect.return_value = inspect_mock

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING
        mock_celery_app.control.inspect.assert_called_with(
            destination=["celery@worker-pod-1"], timeout=2.0
        )

    @patch("dagster_celery.launcher.time.sleep")
    def test_pending_with_hostname_tag_no_ping_reply_returns_unknown(
        self, mock_sleep, launcher, mock_celery_app
    ):
        """An empty ping reply is indistinguishable from a broker/network disruption:
        report UNKNOWN so the monitoring daemon's consecutive-UNKNOWN threshold applies
        instead of immediately failing the run.
        """
        run = _make_run()
        run.tags[DAGSTER_CELERY_WORKER_HOSTNAME_TAG] = "celery@worker-pod-1"
        result = mock_celery_app.AsyncResult.return_value
        result.state = "PENDING"

        inspect_mock = MagicMock()
        inspect_mock.ping.return_value = None
        mock_celery_app.control.inspect.return_value = inspect_mock

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.UNKNOWN

    def test_started_prefers_run_tag_hostname_over_result_meta(self, launcher, mock_celery_app):
        """The run tag is written by the worker that actually executes the run; the
        result meta may have been overwritten by a duplicate delivery on another worker.
        """
        run = _make_run()
        run.tags[DAGSTER_CELERY_WORKER_HOSTNAME_TAG] = "celery@real-worker"
        result = mock_celery_app.AsyncResult.return_value
        result.state = "STARTED"
        result.info = {"hostname": "celery@duplicate-worker"}

        inspect_mock = MagicMock()
        inspect_mock.ping.return_value = {"celery@real-worker": {"ok": "pong"}}
        mock_celery_app.control.inspect.return_value = inspect_mock

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING
        mock_celery_app.control.inspect.assert_called_with(
            destination=["celery@real-worker"], timeout=2.0
        )

    def test_task_failure_returns_failed(self, launcher, mock_celery_app):
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "FAILURE"

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.FAILED

    @patch("dagster_celery.launcher.time.sleep")
    def test_task_pending_retries_then_returns_unknown(self, mock_sleep, launcher, mock_celery_app):
        """PENDING status retries HEALTH_CHECK_MAX_RETRIES times before returning UNKNOWN."""
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "PENDING"

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.UNKNOWN
        assert mock_sleep.call_count == HEALTH_CHECK_MAX_RETRIES - 1

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

    @patch("dagster_celery.launcher.time.sleep")
    def test_task_started_no_ping_reply_retries_then_returns_unknown(
        self, mock_sleep, launcher, mock_celery_app
    ):
        """When the worker doesn't reply to ping, retries then returns UNKNOWN.

        A 2s-timeout broadcast reply cannot distinguish a dead worker from a
        network brownout — a live worker mid-run must not be declared FAILED
        (which would immediately fail the run and strand a zombie task).
        """
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "STARTED"
        result.info = {"hostname": "celery@worker-pod-1"}

        # Worker does NOT respond to ping on any attempt
        inspect_mock = MagicMock()
        inspect_mock.ping.return_value = None
        mock_celery_app.control.inspect.return_value = inspect_mock

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.UNKNOWN
        assert "did not reply to ping" in health.msg.lower()
        assert mock_sleep.call_count == HEALTH_CHECK_MAX_RETRIES - 1

    @patch("dagster_celery.launcher.time.sleep")
    def test_task_started_no_worker_info_retries_then_returns_unknown(
        self, mock_sleep, launcher, mock_celery_app
    ):
        """When worker hostname is unavailable, retries then reports UNKNOWN."""
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "STARTED"
        result.info = None

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.UNKNOWN
        assert "hostname" in health.msg.lower()
        assert mock_sleep.call_count == HEALTH_CHECK_MAX_RETRIES - 1

    @patch("dagster_celery.launcher.time.sleep")
    def test_task_started_inspect_raises_retries_then_returns_unknown(
        self, mock_sleep, launcher, mock_celery_app
    ):
        """When inspect API fails (broker issue), retries then reports UNKNOWN."""
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "STARTED"
        result.info = {"hostname": "celery@worker-pod-1"}

        # Inspect raises an exception on every attempt
        mock_celery_app.control.inspect.side_effect = Exception("Broker connection failed")

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.UNKNOWN
        assert "broker connection failed" in health.msg.lower()
        assert mock_sleep.call_count == HEALTH_CHECK_MAX_RETRIES - 1

    @patch("dagster_celery.launcher.time.sleep")
    def test_ping_fails_then_succeeds_returns_running(self, mock_sleep, launcher, mock_celery_app):
        """If ping fails on first attempts but succeeds on a later one, return RUNNING."""
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "STARTED"
        result.info = {"hostname": "celery@worker-pod-1"}

        # First call: no response. Second call: success.
        inspect_fail = MagicMock()
        inspect_fail.ping.return_value = None
        inspect_ok = MagicMock()
        inspect_ok.ping.return_value = {"celery@worker-pod-1": {"ok": "pong"}}
        mock_celery_app.control.inspect.side_effect = [inspect_fail, inspect_ok]

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING
        assert mock_sleep.call_count == 1

    @patch("dagster_celery.launcher.time.sleep")
    def test_pending_then_started_and_alive_returns_running(
        self, mock_sleep, launcher, mock_celery_app
    ):
        """If task is PENDING on first check but STARTED+alive on retry, return RUNNING."""
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value

        # First call: PENDING. Second call: STARTED with alive worker.
        result_states = iter(["PENDING", "STARTED"])
        type(result).state = property(lambda self, _iter=result_states: next(_iter))
        result.info = {"hostname": "celery@worker-pod-1"}

        inspect_ok = MagicMock()
        inspect_ok.ping.return_value = {"celery@worker-pod-1": {"ok": "pong"}}
        mock_celery_app.control.inspect.return_value = inspect_ok

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING
        assert mock_sleep.call_count == 1


class TestExecuteJobTaskDuplicateDelivery:
    """A broker redelivery of the run worker task must not overwrite the result
    backend state (which run monitoring trusts) with a no-op SUCCESS.
    """

    @pytest.fixture
    def celery_app(self):
        return Celery("test-crash-detection")

    def _run_task(self, celery_app, run_status, hostname="celery@test-worker"):
        task = create_execute_job_task(celery_app)

        mock_instance = MagicMock()
        run = MagicMock()
        run.run_id = "test-run-id"
        run.status = run_status
        mock_instance.get_run_by_id.return_value = run

        with (
            patch("dagster_celery.tasks.DagsterInstance") as mock_instance_cls,
            patch("dagster_celery.tasks.unpack_value") as mock_unpack,
            patch("dagster_celery.tasks._execute_run_command_body") as mock_body,
        ):
            mock_instance_cls.get.return_value.__enter__.return_value = mock_instance
            mock_unpack.return_value = MagicMock(
                run_id="test-run-id", set_exit_code_on_failure=None
            )
            mock_body.return_value = 0

            task.push_request(hostname=hostname)
            try:
                task(execute_job_args_packed={})
            finally:
                task.pop_request()

            return mock_instance, mock_body

    def test_duplicate_delivery_raises_ignore_and_skips_execution(self, celery_app):
        """Run already STARTED means another delivery of this task is (or was) executing
        the run: raise Ignore so celery does not record this delivery as task success.
        """
        with pytest.raises(Ignore):
            self._run_task(celery_app, DagsterRunStatus.STARTED)

    def test_duplicate_delivery_does_not_execute_run(self, celery_app):
        task = create_execute_job_task(celery_app)
        mock_instance = MagicMock()
        run = MagicMock()
        run.status = DagsterRunStatus.STARTED
        mock_instance.get_run_by_id.return_value = run

        with (
            patch("dagster_celery.tasks.DagsterInstance") as mock_instance_cls,
            patch("dagster_celery.tasks.unpack_value") as mock_unpack,
            patch("dagster_celery.tasks._execute_run_command_body") as mock_body,
        ):
            mock_instance_cls.get.return_value.__enter__.return_value = mock_instance
            mock_unpack.return_value = MagicMock(
                run_id="test-run-id", set_exit_code_on_failure=None
            )
            with pytest.raises(Ignore):
                task(execute_job_args_packed={})
            mock_body.assert_not_called()

    def test_first_delivery_executes_and_tags_worker_hostname(self, celery_app):
        mock_instance, mock_body = self._run_task(celery_app, DagsterRunStatus.STARTING)

        mock_body.assert_called_once()
        mock_instance.add_run_tags.assert_called_once_with(
            "test-run-id",
            {DAGSTER_CELERY_WORKER_HOSTNAME_TAG: "celery@test-worker"},
        )
