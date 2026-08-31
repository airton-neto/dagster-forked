from typing import Any, cast

import celery
from celery import Celery
from celery.exceptions import Ignore
from dagster import (
    DagsterInstance,
    DagsterRunStatus,
    _check as check,
)
from dagster._cli.api import _execute_run_command_body, _resume_run_command_body
from dagster._core.definitions.reconstruct import ReconstructableJob
from dagster._core.events import EngineEventData
from dagster._core.execution.api import create_execution_plan, execute_plan_iterator
from dagster._grpc.types import ExecuteRunArgs, ExecuteStepArgs, ResumeRunArgs
from dagster._serdes import serialize_value, unpack_value
from dagster_shared.serdes.serdes import JsonSerializableValue

from dagster_celery.config import (
    TASK_EXECUTE_JOB_NAME,
    TASK_EXECUTE_PLAN_NAME,
    TASK_RESUME_JOB_NAME,
)
from dagster_celery.control import ensure_task_dead
from dagster_celery.core_execution_loop import DELEGATE_MARKER
from dagster_celery.executor import CeleryExecutor
from dagster_celery.tags import DAGSTER_CELERY_WORKER_HOSTNAME_TAG

# Importing the control command registers it on celery's control Panel. Worker
# entry points import this module to build their tasks (the fleet's
# `keda_celery_app.py` does), so every worker gains `ensure_task_dead` without
# any change on their side. Re-exported so the import cannot be tidied away.
__all__ = ["ensure_task_dead"]


def _guard_duplicate_delivery_and_tag_worker(
    instance: DagsterInstance,
    run_id: str,
    hostname: "str | None",
    allowed_statuses: tuple,
) -> None:
    """Raise celery.Ignore for duplicate deliveries of a run worker task.

    A broker redelivery (e.g. visibility timeout on long runs, or worker restart)
    hands the same task to a second worker while the first may still be executing.
    Exiting cleanly would overwrite the result backend state with SUCCESS — which
    run monitoring trusts — so raise Ignore instead, leaving the task state alone.
    Legit deliveries tag the run with the worker hostname so monitoring can ping
    the worker that actually owns the run.
    """
    run = instance.get_run_by_id(run_id)
    if run is not None and run.status not in allowed_statuses:
        instance.report_engine_event(
            f"Ignoring a duplicate Celery delivery of the run worker task (run status:"
            f" {run.status}). The task result state is left untouched so run monitoring"
            " keeps tracking the worker that owns the run.",
            run,
        )
        raise Ignore()
    if run is not None and hostname:
        instance.add_run_tags(run_id, {DAGSTER_CELERY_WORKER_HOSTNAME_TAG: hostname})


def create_task(celery_app, **task_kwargs):
    @celery_app.task(bind=True, name=TASK_EXECUTE_PLAN_NAME, **task_kwargs)
    def _execute_plan(self, execute_step_args_packed, executable_dict):
        execute_step_args = unpack_value(
            check.dict_param(
                execute_step_args_packed,
                "execute_step_args_packed",
            ),
            as_type=ExecuteStepArgs,
        )

        check.dict_param(executable_dict, "executable_dict")

        instance = DagsterInstance.from_ref(execute_step_args.instance_ref)  # pyright: ignore[reportArgumentType]

        recon_job = ReconstructableJob.from_dict(executable_dict)
        retry_mode = execute_step_args.retry_mode

        dagster_run = instance.get_run_by_id(execute_step_args.run_id)
        check.invariant(dagster_run, f"Could not load run {execute_step_args.run_id}")

        step_keys_str = ", ".join(execute_step_args.step_keys_to_execute or [])

        execution_plan = create_execution_plan(
            recon_job,
            dagster_run.run_config,  # pyright: ignore[reportOptionalMemberAccess]
            step_keys_to_execute=execute_step_args.step_keys_to_execute,
            known_state=execute_step_args.known_state,
        )

        engine_event = instance.report_engine_event(
            f"Executing steps {step_keys_str} in celery worker",
            dagster_run,
            EngineEventData(
                {
                    "step_keys": step_keys_str,
                    "Celery worker": self.request.hostname,
                },
                marker_end=DELEGATE_MARKER,
            ),
            CeleryExecutor,
            step_key=execution_plan.step_handle_for_single_step_plans().to_key(),  # pyright: ignore[reportOptionalMemberAccess]
        )

        events = [engine_event]
        for step_event in execute_plan_iterator(
            execution_plan=execution_plan,
            job=recon_job,
            dagster_run=dagster_run,  # pyright: ignore[reportArgumentType]
            instance=instance,
            retry_mode=retry_mode,
            run_config=dagster_run.run_config,  # pyright: ignore[reportOptionalMemberAccess]
        ):
            events.append(step_event)

        serialized_events = [serialize_value(event) for event in events]
        return serialized_events

    return _execute_plan


def _send_to_null(_event: Any) -> None:
    pass


def create_execute_job_task(celery: Celery, **task_kwargs: dict) -> celery.Task:
    """Create a Celery task that executes a run and registers status updates with the
    Dagster instance.
    """

    @celery.task(bind=True, name=TASK_EXECUTE_JOB_NAME, track_started=True, **task_kwargs)
    def _execute_job(_self: Any, execute_job_args_packed: JsonSerializableValue) -> int:
        args: ExecuteRunArgs = unpack_value(
            val=execute_job_args_packed,
            as_type=ExecuteRunArgs,
        )

        with DagsterInstance.get() as instance:
            _guard_duplicate_delivery_and_tag_worker(
                instance=instance,
                run_id=args.run_id,
                hostname=_self.request.hostname,
                allowed_statuses=(DagsterRunStatus.NOT_STARTED, DagsterRunStatus.STARTING),
            )
            return _execute_run_command_body(
                instance=instance,
                run_id=args.run_id,
                write_stream_fn=_send_to_null,
                set_exit_code_on_failure=(
                    args.set_exit_code_on_failure
                    if args.set_exit_code_on_failure is not None
                    else True
                ),
            )

    return cast("celery.Task", _execute_job)


def create_resume_job_task(celery: Celery, **task_kwargs: dict) -> celery.Task:
    """Create a Celery task that resumes a run and registers status updates with the
    Dagster instance.
    """

    @celery.task(bind=True, name=TASK_RESUME_JOB_NAME, track_started=True, **task_kwargs)
    def _resume_job(_self: Any, resume_job_args_packed: JsonSerializableValue) -> int:
        args: ResumeRunArgs = unpack_value(
            val=resume_job_args_packed,
            as_type=ResumeRunArgs,
        )

        with DagsterInstance.get() as instance:
            _guard_duplicate_delivery_and_tag_worker(
                instance=instance,
                run_id=args.run_id,
                hostname=_self.request.hostname,
                allowed_statuses=(DagsterRunStatus.STARTED, DagsterRunStatus.STARTING),
            )
            return _resume_run_command_body(
                instance=instance,
                run_id=args.run_id,
                write_stream_fn=_send_to_null,
                set_exit_code_on_failure=(
                    args.set_exit_code_on_failure
                    if args.set_exit_code_on_failure is not None
                    else True
                ),
            )

    return cast("celery.Task", _resume_job)
