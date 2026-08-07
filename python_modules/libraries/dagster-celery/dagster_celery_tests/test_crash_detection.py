import weakref
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import dagster as dg
import pytest
from celery import Celery
from celery.exceptions import Ignore
from dagster import DagsterRunStatus
from dagster._core.launcher import WorkerStatus
from dagster._core.launcher.base import ResumeRunContext
from dagster._core.test_utils import create_run_for_test, create_test_daemon_workspace_context
from dagster._core.workspace.load_target import EmptyWorkspaceTarget
from dagster._daemon import get_default_daemon_logger
from dagster._daemon.monitoring.run_monitoring import count_resume_run_attempts, monitor_started_run
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
        obj.worker_health_confirmation_cycles = 3
        obj.ping_timeout = 2.0
        obj._worker_health_strikes = {}  # noqa: SLF001
        # Control-plane calls go through a short-lived app; unit tests route them
        # back to the shared mock app.
        obj._control_app = lambda: nullcontext(mock_celery_app)  # noqa: SLF001
        return obj


def _make_run(status=DagsterRunStatus.STARTED, task_id="test-task-123"):
    run = MagicMock()
    run.run_id = "test-run-id"
    run.status = status
    run.tags = {DAGSTER_CELERY_TASK_ID_TAG: task_id}
    run.job_code_origin = MagicMock()
    return run


def _worker_holds_task(inspect_mock, hostname="celery@worker-pod-1", task_id="test-task-123"):
    """Make a mocked inspector report that `hostname` still holds `task_id`.

    A live worker must satisfy the ping AND the query_task identity check, so a
    ping-only stub no longer represents one — see
    ``CeleryRunLauncher._confirm_task_on_worker``.
    """
    inspect_mock.query_task.return_value = {hostname: {task_id: ["active", {}]}}
    return inspect_mock


def _fleet_holds_task(mock_celery_app, task_id="test-task-123"):
    """Make the fleet-wide inventory report that some worker is running `task_id`."""
    inspector = mock_celery_app.control.inspect.return_value
    inspector.active.return_value = {
        "celery@some-worker": [{"id": task_id, "args": "[]", "kwargs": "{}"}]
    }
    inspector.reserved.return_value = {}


def _fleet_empty(mock_celery_app):
    """Workers DO reply, and none of them holds any task.

    Distinct from nobody replying at all: celery returns ``{hostname: []}`` for a
    worker that answered with an empty request table, versus ``None``/``{}`` when
    no worker answered. Only the former is evidence about the task.
    """
    inspector = mock_celery_app.control.inspect.return_value
    inspector.active.return_value = {"celery@worker-pod-1": [], "celery@worker-pod-2": []}
    inspector.reserved.return_value = {"celery@worker-pod-1": [], "celery@worker-pod-2": []}


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

    def test_resume_run_revokes_before_launching(self, launcher, mock_celery_app):
        """The ordering IS the safety property: revoking after launch would leave a
        window with two workers executing the same run.
        """
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

        call_names = [name for name, _args, _kwargs in order_tracker.mock_calls]
        assert call_names.index("revoke") < call_names.index("launch")

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
    def test_pending_with_hostname_tag_no_ping_reply_returns_running_unconfirmed(
        self, mock_sleep, launcher, mock_celery_app
    ):
        """An empty ping reply is indistinguishable from a broker/network disruption:
        report RUNNING (unconfirmed) so stock-core monitoring — which fails runs on any
        non-RUNNING status — takes no action until the strike threshold confirms it.
        """
        run = _make_run()
        run.tags[DAGSTER_CELERY_WORKER_HOSTNAME_TAG] = "celery@worker-pod-1"
        result = mock_celery_app.AsyncResult.return_value
        result.state = "PENDING"

        inspect_mock = MagicMock()
        inspect_mock.ping.return_value = None
        mock_celery_app.control.inspect.return_value = inspect_mock

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING
        assert "unconfirmed" in health.msg.lower()

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
    def test_task_pending_retries_then_returns_running_unconfirmed(
        self, mock_sleep, launcher, mock_celery_app
    ):
        """PENDING retries HEALTH_CHECK_MAX_RETRIES times, then counts a strike."""
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "PENDING"

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING
        assert "unconfirmed" in health.msg.lower()
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
    def test_task_started_no_ping_reply_retries_then_returns_running_unconfirmed(
        self, mock_sleep, launcher, mock_celery_app
    ):
        """When the worker doesn't reply to ping, retries then reports RUNNING (unconfirmed).

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
        assert health.status == WorkerStatus.RUNNING
        assert "did not reply to ping" in health.msg.lower()
        assert mock_sleep.call_count == HEALTH_CHECK_MAX_RETRIES - 1

    @patch("dagster_celery.launcher.time.sleep")
    def test_task_started_no_worker_info_retries_then_returns_running_unconfirmed(
        self, mock_sleep, launcher, mock_celery_app
    ):
        """When worker hostname is unavailable, retries then reports RUNNING (unconfirmed)."""
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "STARTED"
        result.info = None

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING
        assert "hostname" in health.msg.lower()
        assert mock_sleep.call_count == HEALTH_CHECK_MAX_RETRIES - 1

    @patch("dagster_celery.launcher.time.sleep")
    def test_task_started_inspect_raises_retries_then_returns_running_unconfirmed(
        self, mock_sleep, launcher, mock_celery_app
    ):
        """When inspect API fails (broker issue), retries then reports RUNNING (unconfirmed)."""
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "STARTED"
        result.info = {"hostname": "celery@worker-pod-1"}

        # Inspect raises an exception on every attempt
        mock_celery_app.control.inspect.side_effect = Exception("Broker connection failed")

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING
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
        inspect_ok = _worker_holds_task(MagicMock())
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

        inspect_ok = _worker_holds_task(MagicMock())
        inspect_ok.ping.return_value = {"celery@worker-pod-1": {"ok": "pong"}}
        mock_celery_app.control.inspect.return_value = inspect_ok

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING
        assert mock_sleep.call_count == 1


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
        inspect_mock = _worker_holds_task(MagicMock())
        inspect_mock.ping.return_value = {"celery@worker-pod-1": {"ok": "pong"}}
        mock_celery_app.control.inspect.return_value = inspect_mock

    @patch("dagster_celery.launcher.time.sleep")
    def test_strikes_escalate_to_failed_at_threshold(self, _sleep, launcher, mock_celery_app):
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

    @patch("dagster_celery.launcher.time.sleep")
    def test_strikes_reset_after_failed_is_reported(self, _sleep, launcher, mock_celery_app):
        """Once FAILED is reported the streak restarts — a later re-check (e.g. before
        the daemon acts) must not re-fail instantly off stale strikes.
        """
        run = _make_run()
        self._no_ping_reply(mock_celery_app)

        for _ in range(3):
            health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.FAILED

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.RUNNING

    @patch("dagster_celery.launcher.time.sleep")
    def test_healthy_ping_resets_strikes(self, _sleep, launcher, mock_celery_app):
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

    @patch("dagster_celery.launcher.time.sleep")
    def test_strikes_are_tracked_per_run(self, _sleep, launcher, mock_celery_app):
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

    def test_hard_task_failure_needs_no_confirmation(self, launcher, mock_celery_app):
        """Task state FAILURE in the result backend is hard evidence — immediate FAILED."""
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "FAILURE"

        health = launcher.check_run_worker_health(run)
        assert health.status == WorkerStatus.FAILED

    def test_confirmation_cycles_config_default(self):
        field = CeleryRunLauncher.config_type()["worker_health_confirmation_cycles"]
        assert field.default_value == 5


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


class TestIncidentReplayBrokerBrownout:
    """Replay of the 2026-07-28 incident: a short cluster DNS/network brownout made
    inspect.ping() return empty replies for workers that were alive and mid-run. The
    monitoring daemon failed the live run, burned all auto-retries the same way, and
    the never-terminated original task later overwrote FAILURE with SUCCESS.

    These tests wire the real CeleryRunLauncher health check into the real
    monitor_started_run daemon logic (only celery itself is mocked).
    """

    def _setup(self, instance, mock_app, hostname="celery@worker-daily-0"):
        run = create_run_for_test(
            instance,
            job_name="Energia_dos_Ventos_Daily",
            status=DagsterRunStatus.STARTED,
            tags={
                DAGSTER_CELERY_TASK_ID_TAG: "task-1",
                DAGSTER_CELERY_WORKER_HOSTNAME_TAG: hostname,
            },
        )
        result = mock_app.AsyncResult.return_value
        result.state = "STARTED"
        result.info = {"hostname": hostname}
        instance.run_launcher.celery = mock_app
        # Control-plane calls (pings, heartbeat listens) go through a short-lived
        # app; route them to the same mock so only celery itself is mocked.
        instance.run_launcher._control_app = lambda: nullcontext(mock_app)  # noqa: SLF001
        return run, instance.get_run_record_by_id(run.run_id)

    @patch("dagster_celery.launcher.time.sleep")
    def test_short_brownout_does_not_fail_live_run(self, _mock_sleep):
        """A brownout spanning fewer monitoring cycles than the confirmation threshold
        must leave the run untouched, so the live worker finishes it normally.
        """
        mock_app = MagicMock()
        brownout = MagicMock()
        brownout.ping.return_value = {}
        healthy = MagicMock()
        healthy.ping.return_value = {"celery@worker-daily-0": {"ok": "pong"}}

        with (
            dg.instance_for_test(
                overrides={
                    "run_launcher": {
                        "module": "dagster_celery.launcher",
                        "class": "CeleryRunLauncher",
                    },
                    "run_monitoring": {"enabled": True, "max_resume_run_attempts": 1},
                },
            ) as instance,
            create_test_daemon_workspace_context(
                workspace_load_target=EmptyWorkspaceTarget(), instance=instance
            ) as workspace_process_context,
        ):
            logger = get_default_daemon_logger("MonitoringDaemon")
            workspace = workspace_process_context.create_request_context()
            run, run_record = self._setup(instance, mock_app)

            # Two monitoring cycles inside the brownout window (the incident brownout
            # lasted ~3 minutes = at most 2 cycles at the default 120s poll interval)
            mock_app.control.inspect.return_value = brownout
            for _ in range(2):
                monitor_started_run(instance, workspace, run_record, logger)
                current = instance.get_run_by_id(run.run_id)
                assert current is not None
                assert current.status == DagsterRunStatus.STARTED

            # Network recovers; the next cycle sees the worker alive
            mock_app.control.inspect.return_value = healthy
            monitor_started_run(instance, workspace, run_record, logger)

            current = instance.get_run_by_id(run.run_id)
            assert current is not None
            assert current.status == DagsterRunStatus.STARTED
            assert count_resume_run_attempts(instance, run.run_id) == 0
            mock_app.AsyncResult.return_value.revoke.assert_not_called()

    @patch("dagster_celery.launcher.time.sleep")
    def test_sustained_outage_fails_run_and_revokes_task(self, _mock_sleep):
        """When the outage persists past the confirmation threshold and resume attempts
        are exhausted, the run is failed AND its celery task is revoked — a still-alive
        worker can no longer keep executing and overwrite FAILURE with SUCCESS.
        """
        mock_app = MagicMock()
        brownout = MagicMock()
        brownout.ping.return_value = {}

        with (
            dg.instance_for_test(
                overrides={
                    "run_launcher": {
                        "module": "dagster_celery.launcher",
                        "class": "CeleryRunLauncher",
                        "config": {"worker_health_confirmation_cycles": 3},
                    },
                    "run_monitoring": {"enabled": True, "max_resume_run_attempts": 0},
                },
            ) as instance,
            create_test_daemon_workspace_context(
                workspace_load_target=EmptyWorkspaceTarget(), instance=instance
            ) as workspace_process_context,
        ):
            logger = get_default_daemon_logger("MonitoringDaemon")
            workspace = workspace_process_context.create_request_context()
            run, run_record = self._setup(instance, mock_app)

            mock_app.control.inspect.return_value = brownout
            # Cycles 1-2: below the UNKNOWN threshold, run untouched
            for _ in range(2):
                monitor_started_run(instance, workspace, run_record, logger)
                current = instance.get_run_by_id(run.run_id)
                assert current is not None
                assert current.status == DagsterRunStatus.STARTED

            # Cycle 3: threshold reached — run failed and the task revoked
            monitor_started_run(instance, workspace, run_record, logger)
            current = instance.get_run_by_id(run.run_id)
            assert current is not None
            assert current.status == DagsterRunStatus.FAILURE
            mock_app.AsyncResult.return_value.revoke.assert_called_once_with(terminate=True)


def _inventory_task(run_id="test-run-id", task_id="recovered-task-id"):
    """Shape of one entry in celery's inspect().active()/reserved() inventory."""
    return {
        "id": task_id,
        "name": "execute_job",
        "args": "()",
        "kwargs": f"{{'execute_job_args_packed': {{'run_id': '{run_id}', ...}}}}",
    }


class TestMissingTaskIdTag:
    """A run can reach STARTED/STARTING without the celery task id tag when
    ``launch_run`` dies between ``apply_async`` and ``add_run_tags`` (the task was
    submitted, the tag write failed). Health checks must recover the task id from
    the workers' active/reserved inventory — restoring monitorability AND
    revocability — or degrade to the soft-confirmation path instead of raising
    KeyError, which would make the run permanently unmonitorable.
    """

    def _set_inventory(self, mock_celery_app, active=None, reserved=None):
        inspect_mock = MagicMock()
        inspect_mock.active.return_value = active or {}
        inspect_mock.reserved.return_value = reserved or {}
        mock_celery_app.control.inspect.return_value = inspect_mock
        return inspect_mock

    def test_missing_task_id_recovered_from_active_inventory_and_retagged(
        self, launcher, mock_celery_app
    ):
        run = _make_run()
        run.tags = {}
        self._set_inventory(mock_celery_app, active={"celery@worker-pod-1": [_inventory_task()]})
        result = mock_celery_app.AsyncResult.return_value
        result.state = "FAILURE"

        health = launcher.check_run_worker_health(run)

        # The recovered id feeds the NORMAL evidence chain: hard task FAILURE
        # is reported immediately, proving the loop ran with the recovered id.
        assert health.status == WorkerStatus.FAILED
        assert health.msg == "Celery task failed."
        mock_celery_app.AsyncResult.assert_called_once_with("recovered-task-id")
        launcher._instance.add_run_tags.assert_called_once_with(  # noqa: SLF001
            "test-run-id", {DAGSTER_CELERY_TASK_ID_TAG: "recovered-task-id"}
        )

    def test_missing_task_id_recovered_from_reserved_inventory(self, launcher, mock_celery_app):
        run = _make_run()
        run.tags = {}
        self._set_inventory(mock_celery_app, reserved={"celery@worker-pod-1": [_inventory_task()]})
        result = mock_celery_app.AsyncResult.return_value
        result.state = "FAILURE"

        health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.FAILED
        mock_celery_app.AsyncResult.assert_called_once_with("recovered-task-id")

    def test_missing_task_id_retag_failure_still_uses_recovered_id(self, launcher, mock_celery_app):
        run = _make_run()
        run.tags = {}
        self._set_inventory(mock_celery_app, active={"celery@worker-pod-1": [_inventory_task()]})
        launcher._instance.add_run_tags.side_effect = Exception("db down")  # noqa: SLF001
        result = mock_celery_app.AsyncResult.return_value
        result.state = "FAILURE"

        health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.FAILED
        mock_celery_app.AsyncResult.assert_called_once_with("recovered-task-id")

    def test_missing_task_id_other_runs_in_inventory_do_not_match(self, launcher, mock_celery_app):
        run = _make_run()
        run.tags = {}
        self._set_inventory(
            mock_celery_app,
            active={
                "celery@worker-pod-1": [
                    _inventory_task(run_id="some-other-run", task_id="other-task")
                ]
            },
        )

        health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.RUNNING
        assert "unconfirmed (check 1/3)" in health.msg
        assert DAGSTER_CELERY_TASK_ID_TAG in health.msg
        assert "inventory" in health.msg
        mock_celery_app.AsyncResult.assert_not_called()

    def test_missing_task_id_not_in_inventory_escalates_to_failed_at_threshold(
        self, launcher, mock_celery_app
    ):
        run = _make_run()
        run.tags = {}
        self._set_inventory(mock_celery_app)

        first = launcher.check_run_worker_health(run)
        second = launcher.check_run_worker_health(run)
        third = launcher.check_run_worker_health(run)

        assert first.status == WorkerStatus.RUNNING
        assert "unconfirmed (check 1/3)" in first.msg
        assert second.status == WorkerStatus.RUNNING
        assert third.status == WorkerStatus.FAILED
        assert "unconfirmed for 3 consecutive checks" in third.msg
        mock_celery_app.AsyncResult.assert_not_called()

    def test_missing_task_id_inventory_inspect_raises_counts_strike(
        self, launcher, mock_celery_app
    ):
        run = _make_run()
        run.tags = {}
        inspect_mock = MagicMock()
        inspect_mock.active.side_effect = OSError("broker down")
        mock_celery_app.control.inspect.return_value = inspect_mock

        health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.RUNNING
        assert "unconfirmed (check 1/3)" in health.msg
        mock_celery_app.AsyncResult.assert_not_called()

    def test_empty_string_task_id_tag_takes_recovery_path(self, launcher, mock_celery_app):
        """An empty-string tag must not reach AsyncResult("") — it burns the retry
        loop on a bogus PENDING result instead of recovering the real task id.
        """
        run = _make_run(task_id="")
        self._set_inventory(mock_celery_app)

        health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.RUNNING
        assert "unconfirmed (check 1/3)" in health.msg
        mock_celery_app.AsyncResult.assert_not_called()


class TestGetRunWorkerDebugInfo:
    def test_debug_info_with_task_id_reports_task_state(self, launcher, mock_celery_app):
        run = _make_run()
        result = mock_celery_app.AsyncResult.return_value
        result.state = "STARTED"
        result.worker = "celery@worker-pod-1"

        debug_info = launcher.get_run_worker_debug_info(run)

        assert "'celery_task_id': 'test-task-123'" in debug_info
        assert "'task_status': 'STARTED'" in debug_info
        assert "'worker': 'celery@worker-pod-1'" in debug_info

    def test_debug_info_missing_task_id_tag_does_not_raise(self, launcher, mock_celery_app):
        run = _make_run()
        run.tags = {}

        debug_info = launcher.get_run_worker_debug_info(run)

        assert "'celery_task_id': None" in debug_info
        assert "'task_status': None" in debug_info
        assert "test-run-id" in debug_info
        mock_celery_app.AsyncResult.assert_not_called()


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


class TestWorkerIdentityOnHostnameReuse:
    """A ping reply proves the HOSTNAME is alive, not that it still runs OUR task.

    Celery workers run as StatefulSets with stable ordinal hostnames
    (celery@worker-intraday-0). When the pod owning a hostname dies mid-task and
    Kubernetes brings up a replacement — a rolling deploy, a node eviction, or a
    KEDA scale-down-then-up — the new pod answers to the SAME hostname. Trusting
    the ping alone reports a crashed run as RUNNING until max_runtime.

    Migration 0009 fixed this in wheel 0.29.11 with an inspect().query_task()
    identity check; the check was lost in the rebase that produced 0.29.16 and
    stayed missing through .post1/.post2. Observed in production on auren
    2026-08-07: run 1d00a3cf lost its worker at 14:52:16 to a rolling deploy and
    was still reported RUNNING 40+ minutes later, while a sibling run whose
    hostname was genuinely absent was correctly detected and resumed.
    """

    def _started_run_on(self, mock_celery_app, hostname="celery@worker-intraday-0"):
        mock_celery_app.AsyncResult.return_value.state = "STARTED"
        mock_celery_app.control.inspect.return_value.ping.return_value = {hostname: {"ok": "pong"}}
        run = _make_run()
        run.tags = {
            DAGSTER_CELERY_TASK_ID_TAG: "test-task-123",
            DAGSTER_CELERY_WORKER_HOSTNAME_TAG: hostname,
        }
        return run

    def test_impostor_worker_answering_ping_is_not_reported_running(
        self, launcher, mock_celery_app
    ):
        """The replacement pod answers the ping but holds a different task."""
        run = self._started_run_on(mock_celery_app)
        # Hostname is alive, but its request table holds an unrelated task.
        mock_celery_app.control.inspect.return_value.query_task.return_value = {
            "celery@worker-intraday-0": {"some-other-task": ["reserved", {}]}
        }

        health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.FAILED
        assert "no record of" in (health.msg or "")

    def test_worker_still_holding_the_task_is_running(self, launcher, mock_celery_app):
        """The genuine worker lists our task id — healthy, no strike."""
        run = self._started_run_on(mock_celery_app)
        mock_celery_app.control.inspect.return_value.query_task.return_value = {
            "celery@worker-intraday-0": {"test-task-123": ["active", {}]}
        }

        health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.RUNNING
        assert launcher._worker_health_strikes == {}  # noqa: SLF001

    @patch("dagster_celery.launcher.time.sleep")
    def test_query_task_no_reply_is_unconfirmed_not_failed(self, _sleep, launcher, mock_celery_app):
        """Worker answered the ping then went quiet — ambiguous, use the strike path."""
        run = self._started_run_on(mock_celery_app)
        mock_celery_app.control.inspect.return_value.query_task.return_value = {}

        health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.RUNNING
        assert "unconfirmed" in (health.msg or "").lower()

    @patch("dagster_celery.launcher.time.sleep")
    def test_query_task_error_is_unconfirmed_not_failed(self, _sleep, launcher, mock_celery_app):
        """A broker error during the identity check must not fail the run outright."""
        run = self._started_run_on(mock_celery_app)
        mock_celery_app.control.inspect.return_value.query_task.side_effect = OSError(
            "broker connection reset"
        )

        health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.RUNNING
        assert "unconfirmed" in (health.msg or "").lower()


class TestDeathRequiresCorroboration:
    """Silence is not death. Before a run worker is declared dead, a positive
    check must corroborate it — a worker that stops replying to pings, or stops
    logging, may still be executing the task perfectly well (the alupar
    2026-08-04 incident: pings failed 15/15 for days against a live, working
    worker). Every FAILED verdict is therefore gated on a fleet-wide inventory
    that asks all workers whether they still hold the task.
    """

    def _unreachable_tagged_worker(self, mock_celery_app):
        mock_celery_app.AsyncResult.return_value.state = "STARTED"
        inspector = mock_celery_app.control.inspect.return_value
        inspector.ping.return_value = {}
        run = _make_run()
        run.tags = {
            DAGSTER_CELERY_TASK_ID_TAG: "test-task-123",
            DAGSTER_CELERY_WORKER_HOSTNAME_TAG: "celery@worker-pod-1",
        }
        return run

    @patch("dagster_celery.launcher.time.sleep")
    def test_strike_threshold_does_not_fail_a_task_still_running_in_the_fleet(
        self, _sleep, launcher, mock_celery_app
    ):
        """The alupar case: pings lost, but the task is demonstrably still executing."""
        run = self._unreachable_tagged_worker(mock_celery_app)
        _install_heartbeat_events(mock_celery_app, [])
        _fleet_holds_task(mock_celery_app)

        health = None
        for _ in range(launcher.worker_health_confirmation_cycles):
            health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.RUNNING
        assert "still present in a worker's inventory" in (health.msg or "")

    @patch("dagster_celery.launcher.time.sleep")
    def test_strike_threshold_fails_when_fleet_confirms_task_is_gone(
        self, _sleep, launcher, mock_celery_app
    ):
        """Pings lost AND no worker anywhere holds the task — corroborated death."""
        run = self._unreachable_tagged_worker(mock_celery_app)
        _install_heartbeat_events(mock_celery_app, [])
        _fleet_empty(mock_celery_app)

        health = None
        for _ in range(launcher.worker_health_confirmation_cycles):
            health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.FAILED
        assert "no worker in the fleet holds this task" in (health.msg or "")

    def test_hostname_reuse_failure_is_vetoed_when_task_runs_elsewhere(
        self, launcher, mock_celery_app
    ):
        """Task moved to another hostname — the identity mismatch must not kill it."""
        mock_celery_app.AsyncResult.return_value.state = "STARTED"
        inspector = mock_celery_app.control.inspect.return_value
        inspector.ping.return_value = {"celery@worker-pod-1": {"ok": "pong"}}
        inspector.query_task.return_value = {
            "celery@worker-pod-1": {"unrelated-task": ["active", {}]}
        }
        _fleet_holds_task(mock_celery_app)
        run = _make_run()
        run.tags = {
            DAGSTER_CELERY_TASK_ID_TAG: "test-task-123",
            DAGSTER_CELERY_WORKER_HOSTNAME_TAG: "celery@worker-pod-1",
        }

        health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.RUNNING

    @patch("dagster_celery.launcher.time.sleep")
    def test_unreachable_fleet_is_recorded_as_inconclusive(self, _sleep, launcher, mock_celery_app):
        """If nobody replies to the inventory, say so rather than implying proof."""
        run = self._unreachable_tagged_worker(mock_celery_app)
        _install_heartbeat_events(mock_celery_app, [])
        inspector = mock_celery_app.control.inspect.return_value
        inspector.active.return_value = {}
        inspector.reserved.return_value = {}
        inspector.active.side_effect = OSError("broker unreachable")

        health = None
        for _ in range(launcher.worker_health_confirmation_cycles):
            health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.FAILED
        assert "inconclusive" in (health.msg or "")


class FakeHeartbeatReceiver:
    """Stands in for celery.events.Receiver, with kombu's real loop semantics.

    ``ConsumerMixin.consume`` — which ``EventReceiver.capture`` drives — treats its
    ``timeout`` as an *idle* timeout, not a deadline::

        try:
            conn.drain_events(timeout=safety_interval)
        except socket.timeout:
            elapsed += safety_interval
            if timeout and elapsed >= timeout:
                raise
        else:
            yield
            elapsed = 0          # <-- any event resets the budget

    So the loop only ends on ``should_stop`` or on ``timeout`` seconds of *total*
    silence across every worker on the channel. This fake reproduces that: the
    canned events replay forever, and TimeoutError is raised only when there are
    no events at all to deliver.
    """

    # kombu drains with safety_interval=1, so each loop pass costs up to a second.
    SAFETY_INTERVAL = 1.0

    def __init__(self, handlers, events, clock=None):
        self.handlers = handlers
        self.events = events
        self.should_stop = False
        self.iterations = 0
        self._clock = clock

    def on_iteration(self):
        """No-op hook, exactly as kombu defines it — callers may replace it."""

    def _tick(self):
        """One loop pass: time advances, then ConsumerMixin calls on_iteration()."""
        self.iterations += 1
        if self._clock is not None:
            self._clock.advance(self.SAFETY_INTERVAL)
        self.on_iteration()
        if self.iterations > _RUNAWAY_ITERATION_GUARD:
            raise AssertionError(
                "capture() did not terminate: _worker_heartbeat_seen blocked the"
                f" run-monitoring thread for {self.iterations} loop passes."
            )

    def capture(self, limit=None, timeout=None, wakeup=True):
        handler = self.handlers["*"]
        if not self.events:
            # A genuinely idle channel: the built-in idle timeout does fire.
            self._tick()
            if not self.should_stop:
                raise TimeoutError()
            return
        # A busy channel never goes idle, so the loop runs until should_stop.
        while True:
            for event in self.events:
                self._tick()
                if self.should_stop:
                    return
                handler(event)
                if self.should_stop:
                    return


# Bounds the "never terminates" failure mode so the suite fails fast instead of hanging.
_RUNAWAY_ITERATION_GUARD = 5000


class FakeClock:
    """Monotonic clock the fake receiver advances as its loop spins."""

    def __init__(self, now=1000.0):
        self.now = now

    def advance(self, seconds):
        self.now += seconds

    def __call__(self):
        return self.now


def _install_heartbeat_events(mock_celery_app, events, clock=None):
    receivers = []

    def _factory(connection, handlers):
        receiver = FakeHeartbeatReceiver(handlers, events, clock=clock)
        receivers.append(receiver)
        return receiver

    mock_celery_app.events.Receiver = _factory
    return receivers


class TestHeartbeatCorroboration:
    """An empty ping reply alone must not accumulate strikes when the worker is
    demonstrably alive: workers publish heartbeat/task events every few seconds,
    so a fresh event from the tagged hostname is positive evidence of life even
    when the pidbox reply path is broken (alupar 2026-08-04..06 incident: the
    daemon's pings to worker-intraday-0 timed out 15/15 per run for days while
    heartbeats never stopped, killing every intraday run at the 5-strike mark).
    """

    def _started_run_with_empty_ping(self, mock_celery_app, hostname="celery@worker-pod-1"):
        mock_celery_app.AsyncResult.return_value.state = "STARTED"
        mock_celery_app.control.inspect.return_value.ping.return_value = {}
        run = _make_run()
        run.tags = {
            DAGSTER_CELERY_TASK_ID_TAG: "test-task-123",
            DAGSTER_CELERY_WORKER_HOSTNAME_TAG: hostname,
        }
        return run

    def test_empty_ping_with_fresh_heartbeat_returns_running(self, launcher, mock_celery_app):
        run = self._started_run_with_empty_ping(mock_celery_app)
        _install_heartbeat_events(
            mock_celery_app,
            [{"hostname": "celery@worker-pod-1", "type": "worker-heartbeat"}],
        )

        health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.RUNNING
        assert "heartbeat" in (health.msg or "").lower()
        assert launcher._worker_health_strikes == {}  # noqa: SLF001

    def test_empty_ping_with_fresh_heartbeat_does_not_retry_ping(self, launcher, mock_celery_app):
        run = self._started_run_with_empty_ping(mock_celery_app)
        _install_heartbeat_events(
            mock_celery_app,
            [{"hostname": "celery@worker-pod-1", "type": "worker-heartbeat"}],
        )

        launcher.check_run_worker_health(run)

        assert mock_celery_app.control.inspect.call_count == 1

    @patch("dagster_celery.launcher.time.sleep")
    def test_empty_ping_with_foreign_heartbeat_stays_unconfirmed(
        self, _sleep, launcher, mock_celery_app
    ):
        run = self._started_run_with_empty_ping(mock_celery_app)
        _install_heartbeat_events(
            mock_celery_app,
            [{"hostname": "celery@some-other-worker", "type": "worker-heartbeat"}],
        )

        health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.RUNNING
        assert "unconfirmed" in (health.msg or "").lower()
        assert launcher._worker_health_strikes == {"test-run-id": 1}  # noqa: SLF001

    @patch("dagster_celery.launcher.time.sleep")
    def test_empty_ping_with_no_events_stays_unconfirmed(self, _sleep, launcher, mock_celery_app):
        run = self._started_run_with_empty_ping(mock_celery_app)
        _install_heartbeat_events(mock_celery_app, [])

        health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.RUNNING
        assert "unconfirmed" in (health.msg or "").lower()

    @patch("dagster_celery.launcher.time.sleep")
    def test_busy_events_channel_does_not_block_the_monitoring_thread(
        self, _sleep, launcher, mock_celery_app
    ):
        """The heartbeat listen must be bounded by wall clock, not by channel idleness.

        Regression for the 0.29.16.post1 fleet incident: on a server with more than
        one worker, other workers heartbeat every ~2s, so kombu's idle timeout never
        fires. With a dead target worker the listen never returned and the whole
        run-monitoring daemon thread stalled — no max_runtime enforcement, no crash
        detection, no resume, for every run on the server.
        """
        run = self._started_run_with_empty_ping(mock_celery_app, hostname="celery@dead-worker")
        # Foreign traffic only: the target worker is dead, its neighbours are not.
        clock = FakeClock()
        receivers = _install_heartbeat_events(
            mock_celery_app,
            [{"hostname": "celery@some-other-worker", "type": "worker-heartbeat"}],
            clock=clock,
        )

        with patch("dagster_celery.launcher.time.monotonic", clock):
            health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.RUNNING
        assert "unconfirmed" in (health.msg or "").lower()
        assert receivers, "expected the launcher to open an events receiver"
        for receiver in receivers:
            assert receiver.should_stop, (
                "listen must stop itself on a busy channel; leaving should_stop False"
                " is what hung the daemon in production"
            )

    @patch("dagster_celery.launcher.time.sleep")
    def test_heartbeat_listen_stops_at_the_deadline(self, _sleep, launcher, mock_celery_app):
        """The deadline is enforced from the loop hook, so it fires even under traffic."""
        events = [{"hostname": "celery@noisy-neighbour", "type": "worker-heartbeat"}]
        receivers = _install_heartbeat_events(mock_celery_app, events)

        with patch("dagster_celery.launcher.time.monotonic", side_effect=[100.0, 100.0, 999.0]):
            seen = launcher._worker_heartbeat_seen("celery@dead-worker")  # noqa: SLF001

        assert seen is False
        assert receivers[0].should_stop is True

    @patch("dagster_celery.launcher.time.sleep")
    def test_heartbeat_listener_error_falls_back_to_unconfirmed(
        self, _sleep, launcher, mock_celery_app
    ):
        run = self._started_run_with_empty_ping(mock_celery_app)

        def _raise(connection, handlers):
            raise OSError("broker connection refused")

        mock_celery_app.events.Receiver = _raise

        health = launcher.check_run_worker_health(run)

        assert health.status == WorkerStatus.RUNNING
        assert "unconfirmed" in (health.msg or "").lower()


class TestPingTimeout:
    def test_ping_uses_configured_timeout(self, launcher, mock_celery_app):
        launcher.ping_timeout = 7.5
        mock_celery_app.control.inspect.return_value.ping.return_value = {
            "celery@worker-pod-1": {"ok": "pong"}
        }

        result = launcher._ping_hostname("celery@worker-pod-1")  # noqa: SLF001

        assert result.status == WorkerStatus.RUNNING
        mock_celery_app.control.inspect.assert_called_once_with(
            destination=["celery@worker-pod-1"], timeout=7.5
        )

    def test_ping_timeout_config_field_default(self):
        field = CeleryRunLauncher.config_type()["ping_timeout"]
        assert field.default_value == 10.0
        assert not field.is_required


class TestFreshControlApp:
    def test_ping_goes_through_control_app_not_cached_app(self, launcher, mock_celery_app):
        """Pings must use the short-lived control app, never the long-lived cached
        app: the cached app's mailbox reply routing can rot (alupar incident),
        while a fresh app per check matches the always-healthy CLI behavior.
        """
        sentinel_app = MagicMock()
        sentinel_app.control.inspect.return_value.ping.return_value = {
            "celery@worker-pod-1": {"ok": "pong"}
        }
        launcher._control_app = lambda: nullcontext(sentinel_app)  # noqa: SLF001

        result = launcher._ping_hostname("celery@worker-pod-1")  # noqa: SLF001

        assert result.status == WorkerStatus.RUNNING
        sentinel_app.control.inspect.assert_called_once()
        mock_celery_app.control.inspect.assert_not_called()

    def test_default_control_app_builds_and_closes_fresh_app(self):
        with patch.object(CeleryRunLauncher, "__init__", lambda self: None):
            obj = CeleryRunLauncher.__new__(CeleryRunLauncher)
        obj.broker = "redis://localhost:6379/0"
        obj.backend = "redis://localhost:6379/0"
        obj.include = []
        obj.config_source = {}
        obj.default_queue = "dagster"

        with patch("dagster_celery.launcher.make_app") as mock_make_app:
            fresh = MagicMock()
            mock_make_app.return_value = fresh
            with obj._control_app() as app:  # noqa: SLF001
                assert app is fresh
                fresh.close.assert_not_called()
            mock_make_app.assert_called_once_with(app_args=obj.app_args())
            fresh.close.assert_called_once()

    def test_recover_task_id_goes_through_control_app(self, launcher, mock_celery_app):
        sentinel_app = MagicMock()
        sentinel_app.control.inspect.return_value.active.return_value = {}
        sentinel_app.control.inspect.return_value.reserved.return_value = {}
        launcher._control_app = lambda: nullcontext(sentinel_app)  # noqa: SLF001
        run = _make_run(task_id=None)
        run.tags = {}

        launcher._recover_task_id(run)  # noqa: SLF001

        sentinel_app.control.inspect.assert_called_once()
        mock_celery_app.control.inspect.assert_not_called()
