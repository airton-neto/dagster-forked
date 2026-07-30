import logging
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from celery import Celery
from dagster import (
    DagsterInstance,
    DagsterRun,
    Field,
    Noneable,
    Permissive,
    StringSource,
    _check as check,
)
from dagster._core.events import EngineEventData
from dagster._core.launcher import (
    CheckRunHealthResult,
    LaunchRunContext,
    ResumeRunContext,
    RunLauncher,
    WorkerStatus,
)
from dagster._grpc.types import ExecuteRunArgs, ResumeRunArgs
from dagster._serdes import ConfigurableClass, ConfigurableClassData, pack_value
from typing_extensions import Self, override

# Retry constants for health check — absorb transient broker/inspect failures
# before reporting UNKNOWN or FAILED to the stock monitoring daemon.
HEALTH_CHECK_MAX_RETRIES = 3
HEALTH_CHECK_RETRY_DELAY_SECONDS = 5.0

# Task state SUCCESS with the run still unfinished is only tolerated this long
# (the run terminal event is normally written before the task returns).
TASK_SUCCESS_TERMINAL_GRACE_SECONDS = 60.0

# Soft (unconfirmed) health outcomes — empty ping reply, PENDING, broker errors —
# must repeat for this many consecutive monitoring cycles before FAILED is
# reported. Stock dagster core acts immediately on any non-RUNNING/SUCCESS
# status, so this confirmation MUST live in the launcher: unconfirmed cycles
# report RUNNING (treat-as-alive) instead of UNKNOWN.
DEFAULT_WORKER_HEALTH_CONFIRMATION_CYCLES = 5

# Backstop against unbounded strike-dict growth (entries for runs that finished
# mid-streak are never individually cleaned; the daemon process restart also
# clears them).
_MAX_TRACKED_HEALTH_STRIKES = 1000

from dagster_celery.config import DEFAULT_CONFIG, TASK_EXECUTE_JOB_NAME, TASK_RESUME_JOB_NAME
from dagster_celery.defaults import task_default_queue
from dagster_celery.make_app import make_app
from dagster_celery.tags import (
    DAGSTER_CELERY_QUEUE_TAG,
    DAGSTER_CELERY_RUN_PRIORITY_TAG,
    DAGSTER_CELERY_TASK_ID_TAG,
    DAGSTER_CELERY_WORKER_HOSTNAME_TAG,
)
from dagster_celery.tasks import create_execute_job_task, create_resume_job_task

if TYPE_CHECKING:
    from celery.result import AsyncResult
    from dagster._config import UserConfigSchema


class CeleryRunLauncher(RunLauncher, ConfigurableClass):
    """Dagster [Run Launcher](https://docs.dagster.io/guides/deploy/execution/run-launchers) which
    starts runs as Celery tasks.

    Supports run worker crash detection and automatic resume. When ``run_monitoring``
    is enabled in ``dagster.yaml``, the daemon periodically calls
    ``check_run_worker_health`` which pings the Celery worker via the
    ``inspect`` API. If the worker is unreachable the run is marked as
    failed and, when ``max_resume_run_attempts > 0``, resumed on a
    healthy worker.

    Requires a persistent result backend (e.g. Redis) so that task state
    survives worker restarts.
    """

    _instance: DagsterInstance  # pyright: ignore[reportIncompatibleMethodOverride]
    celery: Celery

    def __init__(
        self,
        default_queue: str,
        broker: str | None = None,
        backend: str | None = None,
        include: list[str] | None = None,
        config_source: dict | None = None,
        inst_data: ConfigurableClassData | None = None,
        worker_health_confirmation_cycles: int | None = None,
    ) -> None:
        self._inst_data = check.opt_inst_param(inst_data, "inst_data", ConfigurableClassData)

        self.broker = check.opt_str_param(broker, "broker", default=broker)
        self.backend = check.opt_str_param(backend, "backend", default=backend)
        self.include = check.opt_list_param(include, "include", of_type=str)
        self.config_source = dict(
            DEFAULT_CONFIG, **check.opt_dict_param(config_source, "config_source")
        )
        self.default_queue = check.str_param(default_queue, "default_queue")
        self.worker_health_confirmation_cycles = check.opt_int_param(
            worker_health_confirmation_cycles,
            "worker_health_confirmation_cycles",
            default=DEFAULT_WORKER_HEALTH_CONFIRMATION_CYCLES,
        )
        # Consecutive soft-failure strikes per run_id, held in the (long-lived)
        # monitoring daemon process.
        self._worker_health_strikes: dict[str, int] = {}

        self.celery = make_app(
            app_args=self.app_args(),
        )

        if backend and backend.startswith("rpc://"):
            logging.getLogger(__name__).warning(
                "CeleryRunLauncher is configured with the 'rpc://' result backend. "
                "Crash detection via worker ping requires a persistent result backend "
                "(e.g. Redis). With 'rpc://', task state is lost when a worker crashes "
                "and monitoring falls back to the PENDING/UNKNOWN detection path."
            )

        super().__init__()

    def app_args(self) -> dict:
        return {
            "broker": self.broker,
            "backend": self.backend,
            "include": self.include,
            "config_source": self.config_source,
            "task_default_queue": self.default_queue,
        }

    def launch_run(self, context: LaunchRunContext) -> None:
        run = context.dagster_run
        job_origin = check.not_none(run.job_code_origin)

        args = ExecuteRunArgs(
            job_origin=job_origin,
            run_id=run.run_id,
            instance_ref=self._instance.get_ref(),
            set_exit_code_on_failure=True,
        )

        task = create_execute_job_task(self.celery)
        task_signature = task.si(
            execute_job_args_packed=pack_value(args),
        )

        self._launch_celery_task_run(
            run=run,
            task_signature=task_signature,
            routing_key=TASK_EXECUTE_JOB_NAME,
        )

    def terminate(self, run_id: str) -> bool:
        run = self._instance.get_run_by_id(run_id)
        if run is None:
            return False

        # Deliberately NO `run.is_finished` guard (unlike other launchers): run
        # monitoring calls this AFTER marking the run failed, precisely to revoke
        # the celery task of a worker that may still be alive and executing.
        # Adding the guard would silently disable the zombie-worker protection.
        task_id = run.tags.get(DAGSTER_CELERY_TASK_ID_TAG)
        if task_id is None:
            return False

        result: AsyncResult = self.celery.AsyncResult(task_id)
        result.revoke(terminate=True)

        return True

    @property
    def supports_resume_run(self) -> bool:
        return True

    def resume_run(self, context: ResumeRunContext) -> None:
        run = context.dagster_run
        job_origin = check.not_none(run.job_code_origin)

        # The prior worker may still be alive — health checks can false-positive
        # during a broker/network brownout — and two workers must not execute the
        # same run concurrently. Best-effort: the revoke broadcast may not reach a
        # worker that is currently partitioned from the broker.
        prior_task_id = run.tags.get(DAGSTER_CELERY_TASK_ID_TAG)
        if prior_task_id:
            try:
                prior_result: AsyncResult = self.celery.AsyncResult(prior_task_id)
                prior_result.revoke(terminate=True)
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to revoke prior Celery task %s before resuming run %s.",
                    prior_task_id,
                    run.run_id,
                    exc_info=True,
                )

        args = ResumeRunArgs(
            job_origin=job_origin,
            run_id=run.run_id,
            instance_ref=self._instance.get_ref(),
            set_exit_code_on_failure=True,
        )

        task = create_resume_job_task(self.celery)
        task_signature = task.si(
            resume_job_args_packed=pack_value(args),
        )

        self._launch_celery_task_run(
            run=run,
            task_signature=task_signature,
            routing_key=TASK_RESUME_JOB_NAME,
        )

    def _launch_celery_task_run(
        self,
        run: DagsterRun,
        task_signature: Celery.Task,
        routing_key: str,
    ) -> None:
        run_priority = _get_run_priority(run)
        queue = run.tags.get(DAGSTER_CELERY_QUEUE_TAG, self.default_queue)

        self._instance.report_engine_event(
            "Creating Celery run worker job task",
            run,
            cls=self.__class__,
        )

        result: AsyncResult = task_signature.apply_async(
            priority=run_priority,
            queue=queue,
            routing_key=f"{queue}.{routing_key}",
        )

        self._instance.add_run_tags(
            run.run_id,
            {DAGSTER_CELERY_TASK_ID_TAG: result.task_id},
        )

        self._instance.report_engine_event(
            "Celery task has been forwarded to the broker.",
            run,
            EngineEventData(
                {
                    "Run ID": run.run_id,
                    "Celery Task ID": result.task_id,
                    "Celery Queue": queue,
                }
            ),
            cls=self.__class__,
        )

    @property
    def supports_check_run_worker_health(self) -> bool:
        return True

    def check_run_worker_health(self, run: DagsterRun) -> CheckRunHealthResult:
        """Check whether the Celery worker running this task is alive.

        Hard evidence (task state FAILURE, task SUCCESS without a run terminal
        event) is reported as FAILED immediately. Soft outcomes — empty ping
        reply, PENDING, broker errors — are indistinguishable from a transient
        broker/network disruption, so they report RUNNING (treat-as-alive) until
        `worker_health_confirmation_cycles` consecutive monitoring cycles agree,
        and only then FAILED. Stock dagster core acts on any non-RUNNING/SUCCESS
        status immediately, which is why the confirmation lives here rather than
        in the monitoring daemon.
        """
        raw = self._check_run_worker_health_raw(run)
        return self._confirm_worker_health(run.run_id, raw)

    def _confirm_worker_health(
        self, run_id: str, raw: CheckRunHealthResult
    ) -> CheckRunHealthResult:
        if len(self._worker_health_strikes) > _MAX_TRACKED_HEALTH_STRIKES:
            self._worker_health_strikes.clear()

        if raw.status in (WorkerStatus.RUNNING, WorkerStatus.SUCCESS, WorkerStatus.FAILED):
            # RUNNING/SUCCESS: healthy — reset the streak. FAILED: hard evidence
            # from the result backend — no confirmation needed.
            self._worker_health_strikes.pop(run_id, None)
            return raw

        strikes = self._worker_health_strikes.get(run_id, 0) + 1
        if strikes >= self.worker_health_confirmation_cycles:
            self._worker_health_strikes.pop(run_id, None)
            return CheckRunHealthResult(
                WorkerStatus.FAILED,
                f"Worker health unconfirmed for {strikes} consecutive checks: {raw.msg}",
            )

        self._worker_health_strikes[run_id] = strikes
        return CheckRunHealthResult(
            WorkerStatus.RUNNING,
            f"Worker health unconfirmed"
            f" (check {strikes}/{self.worker_health_confirmation_cycles}), treating as"
            f" alive: {raw.msg}",
        )

    def _check_run_worker_health_raw(self, run: DagsterRun) -> CheckRunHealthResult:
        """Single-cycle health probe.

        A run missing the celery task id tag short-circuits into task-id recovery
        from the worker inventory (see ``_recover_task_id``) before the retry loop;
        unrecoverable runs report UNKNOWN. Transient failures (PENDING/broker
        errors) retry up to HEALTH_CHECK_MAX_RETRIES times before reporting
        UNKNOWN, which `_confirm_worker_health` then absorbs into the strike
        counter.
        """
        logger = logging.getLogger(__name__)
        task_id = run.tags.get(DAGSTER_CELERY_TASK_ID_TAG)
        if not task_id:
            # launch_run died between apply_async and add_run_tags (or the tag is
            # empty): the task may be executing, but the result backend cannot be
            # queried without its id. Recover the id from the workers' active and
            # reserved task inventories and re-tag the run — restoring both
            # monitorability and revocability. When the run is in no inventory the
            # task is not executing anywhere reachable: report UNKNOWN and let the
            # strike counter decide (FAILED after N cycles frees monitoring to
            # resume the run; a stale broker-queued duplicate of the original task
            # is rejected by the delivery guard in tasks.py).
            task_id = self._recover_task_id(run)
            if not task_id:
                return CheckRunHealthResult(
                    WorkerStatus.UNKNOWN,
                    f"Run has no {DAGSTER_CELERY_TASK_ID_TAG} tag and was not found in"
                    " any worker's active/reserved task inventory — the celery task id"
                    " was never recorded (launch likely failed after task submission).",
                )

        last_result: CheckRunHealthResult | None = None
        for attempt in range(1, HEALTH_CHECK_MAX_RETRIES + 1):
            result: AsyncResult = self.celery.AsyncResult(task_id)
            task_status = result.state

            if task_status == "SUCCESS":
                return self._check_task_success_run_terminal(run, result)
            if task_status == "FAILURE":
                return CheckRunHealthResult(WorkerStatus.FAILED, "Celery task failed.")
            if task_status == "STARTED":
                ping_result = self._ping_worker(run, result)
                if ping_result.status == WorkerStatus.RUNNING:
                    return ping_result
                # Ping failed — might be transient, retry
                last_result = ping_result
            else:
                # PENDING, RETRYING, etc. PENDING may mean the result backend lost
                # the task meta (e.g. redis restart); if the run worker tagged the
                # run with its hostname, ping it directly instead of giving up.
                tagged_hostname = run.tags.get(DAGSTER_CELERY_WORKER_HOSTNAME_TAG)
                if tagged_hostname:
                    ping_result = self._ping_hostname(tagged_hostname)
                    if ping_result.status == WorkerStatus.RUNNING:
                        return ping_result
                    last_result = CheckRunHealthResult(
                        ping_result.status,
                        f"Task status {task_status}; {ping_result.msg}",
                    )
                else:
                    last_result = CheckRunHealthResult(
                        WorkerStatus.UNKNOWN, f"Unknown task status: {task_status}"
                    )

            if attempt < HEALTH_CHECK_MAX_RETRIES:
                logger.info(
                    "Health check attempt %d/%d for run %s returned %s — retrying in %.0fs. %s",
                    attempt,
                    HEALTH_CHECK_MAX_RETRIES,
                    run.run_id,
                    last_result.status if last_result else "N/A",
                    HEALTH_CHECK_RETRY_DELAY_SECONDS,
                    last_result.msg if last_result else "",
                )
                time.sleep(HEALTH_CHECK_RETRY_DELAY_SECONDS)

        logger.warning(
            "Health check for run %s exhausted %d retries. Final status: %s — %s",
            run.run_id,
            HEALTH_CHECK_MAX_RETRIES,
            last_result.status if last_result else "N/A",
            last_result.msg if last_result else "",
        )
        return last_result  # type: ignore[return-value]

    def _recover_task_id(self, run: DagsterRun) -> "str | None":
        """Find the celery task executing this run when the task id tag is missing.

        Broadcasts an active/reserved inventory request to all workers and matches
        the run id inside the task arguments. On a match the run is re-tagged so
        subsequent health checks, ``terminate`` and ``resume_run`` can address the
        task again; the recovered id is still used for the current check when the
        tag write fails.
        """
        logger = logging.getLogger(__name__)
        try:
            inspector = self.celery.control.inspect(timeout=2.0)
            inventories = (inspector.active() or {}, inspector.reserved() or {})
        except Exception as e:
            logger.warning(
                "Failed to inspect the worker task inventory while recovering the"
                " celery task id for run %s: %s",
                run.run_id,
                e,
            )
            return None

        for inventory in inventories:
            for tasks in inventory.values():
                for task in tasks or ():
                    if run.run_id not in f"{task.get('args')}{task.get('kwargs')}":
                        continue
                    task_id = task.get("id")
                    if not task_id:
                        continue
                    try:
                        self._instance.add_run_tags(
                            run.run_id, {DAGSTER_CELERY_TASK_ID_TAG: task_id}
                        )
                    except Exception:
                        logger.warning(
                            "Recovered celery task id %s for run %s but re-tagging"
                            " failed; using it for this check only.",
                            task_id,
                            run.run_id,
                            exc_info=True,
                        )
                    logger.info(
                        "Recovered celery task id %s for run %s from the worker task inventory.",
                        task_id,
                        run.run_id,
                    )
                    return task_id
        return None

    def _check_task_success_run_terminal(
        self, run: DagsterRun, result: "AsyncResult"
    ) -> CheckRunHealthResult:
        """Task state SUCCESS is only healthy if the run reached a terminal state.

        A duplicate (redelivered) execution of the run worker task exits as a no-op
        and overwrites the result backend with SUCCESS while the run is still in
        flight; trusting it unconditionally leaves crashed runs STARTED forever.
        """
        current_run = self._instance.get_run_by_id(run.run_id)
        if current_run is None or current_run.is_finished:
            return CheckRunHealthResult(WorkerStatus.SUCCESS)

        date_done = result.date_done
        if date_done is not None:
            if date_done.tzinfo is None:
                date_done = date_done.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - date_done).total_seconds()
            if elapsed < TASK_SUCCESS_TERMINAL_GRACE_SECONDS:
                return CheckRunHealthResult(
                    WorkerStatus.RUNNING,
                    "Celery task succeeded; waiting for the run terminal event.",
                )

        return CheckRunHealthResult(
            WorkerStatus.FAILED,
            f"Celery task {result.id} reports SUCCESS but run {run.run_id} never reached a"
            f" terminal state (status: {current_run.status}). The task result was likely"
            " overwritten by a duplicate delivery, or the worker exited without finalizing"
            " the run.",
        )

    def _ping_worker(self, run: DagsterRun, result: "AsyncResult") -> CheckRunHealthResult:
        """Ping the Celery worker for a STARTED task. Single attempt, no retries.

        Prefers the hostname tagged on the run by the executing worker over the
        result-backend meta, which a duplicate delivery may have overwritten.
        """
        logger = logging.getLogger(__name__)
        worker_hostname = run.tags.get(
            DAGSTER_CELERY_WORKER_HOSTNAME_TAG
        ) or self._get_worker_hostname(result)
        if not worker_hostname:
            logger.warning(
                "Cannot determine Celery worker hostname from task result. "
                "Reporting worker status as UNKNOWN."
            )
            return CheckRunHealthResult(
                WorkerStatus.UNKNOWN,
                "Cannot determine Celery worker hostname from task result.",
            )

        return self._ping_hostname(worker_hostname)

    def _ping_hostname(self, worker_hostname: str) -> CheckRunHealthResult:
        logger = logging.getLogger(__name__)
        try:
            inspector = self.celery.control.inspect(
                destination=[worker_hostname],
                timeout=2.0,
            )
            ping_response = inspector.ping()
        except Exception as e:
            logger.warning(
                "Failed to ping Celery worker %s: %s. Reporting worker status as UNKNOWN.",
                worker_hostname,
                e,
            )
            return CheckRunHealthResult(
                WorkerStatus.UNKNOWN,
                f"Failed to ping Celery worker {worker_hostname}: {e}",
            )
        if ping_response and isinstance(ping_response, dict) and worker_hostname in ping_response:
            return CheckRunHealthResult(WorkerStatus.RUNNING)

        # An empty reply is indistinguishable from a broker/network brownout: the
        # broadcast reply simply may not have arrived within the timeout while the
        # worker is alive and mid-run. Report UNKNOWN so the monitoring daemon's
        # consecutive-UNKNOWN threshold decides, instead of failing the run on a
        # single lost ping (which strands a zombie worker that later overwrites the
        # FAILURE status with SUCCESS).
        return CheckRunHealthResult(
            WorkerStatus.UNKNOWN,
            f"Celery worker {worker_hostname} did not reply to ping within the timeout.",
        )

    @staticmethod
    def _get_worker_hostname(result: "AsyncResult") -> "str | None":
        """Extract the worker hostname from an AsyncResult.

        The hostname is available via result.info when the task has been started,
        or via result.worker on some backends.
        """
        # Try result.info dict (standard approach)
        info = result.info
        if isinstance(info, dict) and "hostname" in info:
            return info["hostname"]

        # Try result.worker attribute (available on some backends)
        worker = getattr(result, "worker", None)
        if isinstance(worker, str) and worker:
            return worker

        return None

    @override
    def get_run_worker_debug_info(
        self, run: DagsterRun, include_container_logs: bool | None = True
    ) -> str | None:
        task_id = run.tags.get(DAGSTER_CELERY_TASK_ID_TAG)

        task_status = None
        worker = None
        if task_id:
            result: AsyncResult = self.celery.AsyncResult(task_id)
            task_status = result.state
            worker = result.worker

        return str(
            {
                "run_id": run.run_id,
                "celery_task_id": task_id,
                "task_status": task_status,
                "worker": worker,
            }
        )

    @property
    def inst_data(self) -> ConfigurableClassData | None:
        return self._inst_data

    @classmethod
    def config_type(cls) -> "UserConfigSchema":
        return {
            "broker": Field(
                Noneable(StringSource),
                is_required=False,
                description=(
                    "The URL of the Celery broker. Default: "
                    "'pyamqp://guest@{os.getenv('DAGSTER_CELERY_BROKER_HOST',"
                    "'localhost')}//'."
                ),
            ),
            "backend": Field(
                Noneable(StringSource),
                is_required=False,
                default_value="rpc://",
                description=(
                    "The URL of the Celery results backend. Default: 'rpc://'. "
                    "Note: crash detection via worker ping requires a persistent "
                    "backend such as Redis (e.g. 'redis://localhost:6379/0'). "
                    "The default 'rpc://' backend loses task state on worker "
                    "crash, falling back to the PENDING/UNKNOWN detection path."
                ),
            ),
            "include": Field(
                [str],
                is_required=False,
                description="List of modules every worker should import",
            ),
            "default_queue": Field(
                StringSource,
                is_required=False,
                description="The default queue to use when a run does not specify "
                "Celery queue tag.",
                default_value=task_default_queue,
            ),
            "config_source": Field(
                Noneable(Permissive()),
                is_required=False,
                description="Additional settings for the Celery app.",
            ),
            "worker_health_confirmation_cycles": Field(
                int,
                is_required=False,
                default_value=DEFAULT_WORKER_HEALTH_CONFIRMATION_CYCLES,
                description=(
                    "Consecutive monitoring cycles a soft worker-health failure (empty"
                    " ping reply, PENDING task state, broker errors) must persist before"
                    " the run worker is reported FAILED. Unconfirmed cycles report the"
                    " worker as alive, absorbing transient broker/network disruptions."
                ),
            ),
        }

    @classmethod
    def from_config_value(
        cls, inst_data: ConfigurableClassData, config_value: Mapping[str, Any]
    ) -> Self:
        return cls(inst_data=inst_data, **config_value)


def _get_run_priority(run: DagsterRun) -> int:
    if DAGSTER_CELERY_RUN_PRIORITY_TAG not in run.tags:
        return 0
    try:
        return int(run.tags[DAGSTER_CELERY_RUN_PRIORITY_TAG])
    except ValueError:
        return 0
