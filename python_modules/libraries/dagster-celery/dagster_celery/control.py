"""Worker-side control command that confirms a revoked task actually died.

``AsyncResult.revoke(terminate=True)`` sends SIGTERM to the billiard child.
Dagster's op-concurrency-pool wait loop does not check the captured interrupt,
so a step that is waiting on a pool absorbs the signal and keeps its Celery slot
until the step finally starts — 14 hours later in the auren-aes 2026-08-31
incident, where six such zombies wedged an intraday fleet of ``concurrency=1``
workers.

The confirmation cannot live in the launcher. ``Request.terminate`` calls
``_announce_revoked`` -> ``celery.worker.state.task_ready``, which pops the
request from ``requests`` and discards it from ``active_requests`` and
``reserved_requests`` the moment SIGTERM is sent, whether or not the child dies
(``celery/worker/request.py:413-417``, ``celery/worker/state.py:118-129``).
So:

- the fleet inventory reports the task gone immediately, and no amount of
  polling from the launcher can observe the zombie, and
- a second ``revoke`` finds nothing, because ``_revoke`` resolves targets
  through ``_find_requests_by_id`` over that same emptied table
  (``celery/worker/control.py:115-121,228``).

The billiard ``ApplyResult`` in the pool cache outlives ``task_ready``: it is
keyed by job number in ``pool._cache`` (``billiard/pool.py:1748``) and still
references its ``Request`` through the ``accept_callback`` bound method that
celery passes as ``self.on_accepted`` (``celery/worker/request.py:362``). That
gives the worker a path from a task id to the pid to kill.

Traced against celery 5.6.2 and billiard 4.2.4. Every attribute reached here is
private, so each hop is guarded and any failure degrades to "do nothing" — a
control handler that raises would take down the worker's consumer.

Caveat: SIGKILL to the pool child does not reap grandchildren. A task using the
multiprocess executor can leak its step subprocesses. Our fleet runs
``in_process_executor`` inside the Celery task, so the child is the whole run.
"""

import logging
import signal
import threading

from celery.worker.control import control_command

# How long the worker waits after SIGTERM before it kills the child. Internal by
# design: this is a correctness guarantee of `CeleryRunLauncher.terminate`, not
# an operator choice, so it carries no config field. The launcher passes it as
# the `grace_seconds` broadcast argument.
TERMINATE_GRACE_SECONDS = 20.0

ENSURE_TASK_DEAD_COMMAND = "ensure_task_dead"

# Upper bound for a caller-supplied grace window. A broadcast argument is
# attacker-adjacent input as far as the worker is concerned, and a timer thread
# parked for hours is a leak.
MAX_GRACE_SECONDS = 300.0


def _pool_cache(state):
    """The billiard pool's job cache, or None for a pool that has no such thing.

    Only the prefork pool exposes ``_pool._cache``; solo and threads pools do
    not, and a worker still starting up has no pool at all.
    """
    pool = getattr(state.consumer, "pool", None)
    if pool is None:
        return None
    inner = getattr(pool, "_pool", None)
    if inner is None:
        return None
    return getattr(inner, "_cache", None)


def _task_id_of(result) -> str | None:
    """Read the task id an ``ApplyResult`` belongs to.

    Two independent paths, because both are private: celery sets
    ``correlation_id=task_id`` when it submits the job, and the
    ``accept_callback`` is the ``Request.on_accepted`` bound method, so its
    ``__self__`` is the ``Request``.
    """
    correlation_id = getattr(result, "correlation_id", None)
    if correlation_id:
        return correlation_id

    accept_callback = getattr(result, "_accept_callback", None)
    request = getattr(accept_callback, "__self__", None)
    return getattr(request, "id", None)


def _find_live_result(state, task_id: str):
    """The pool-cache entry for ``task_id`` on THIS worker, or None.

    Returns None when the task belongs to another worker, already finished, or
    has no pid yet. Never raises.
    """
    logger = logging.getLogger(__name__)
    try:
        cache = _pool_cache(state)
        if not cache:
            return None

        for result in list(cache.values()):
            try:
                if _task_id_of(result) != task_id:
                    continue
                if result.ready():
                    return None
                if not getattr(result, "_worker_pid", None):
                    return None
                return result
            except Exception:
                # One unusable cache entry must not hide the task behind it.
                logger.debug("ensure_task_dead: skipping unreadable cache entry", exc_info=True)
                continue
        return None
    except Exception:
        logger.warning("ensure_task_dead: cache lookup failed for task %s", task_id, exc_info=True)
        return None


def _coerce_grace_seconds(grace_seconds) -> float:
    """Clamp a broadcast-supplied grace window into [0, MAX_GRACE_SECONDS].

    ``@control_command(args=...)`` declares types but validates nothing, so the
    value arriving here is whatever the publisher put on the wire.
    """
    try:
        value = float(grace_seconds)
    except (TypeError, ValueError):
        return TERMINATE_GRACE_SECONDS
    if value != value:  # NaN
        return TERMINATE_GRACE_SECONDS
    return max(0.0, min(value, MAX_GRACE_SECONDS))


def _ensure_task_dead_now(state, task_id: str) -> bool:
    """Kill the pool child still running ``task_id``, if there is one.

    Returns True when SIGKILL was sent. Returns False in every other case —
    task already gone, result already ready, pid unknown, unusable pool, or any
    error along the way. Never raises.

    Residual race: the job can finish between the final ``ready()`` check and
    ``terminate_job``, and with ``max-tasks-per-child`` the same pid can already
    be running the NEXT task, which then dies instead. The re-check below shrinks
    that window to microseconds. If it is ever lost, the innocent task raises
    ``WorkerLostError``, celery redelivers it (``task_acks_late``), and the
    duplicate-delivery guard in ``tasks.py`` keeps the redelivery safe.
    """
    logger = logging.getLogger(__name__)
    try:
        cache = _pool_cache(state)
        if not cache:
            logger.debug("ensure_task_dead: no usable pool cache for task %s", task_id)
            return False

        for result in list(cache.values()):
            try:
                if _task_id_of(result) != task_id:
                    continue
                if result.ready():
                    logger.debug("ensure_task_dead: task %s already finished", task_id)
                    return False
                worker_pid = getattr(result, "_worker_pid", None)
                if not worker_pid:
                    logger.debug("ensure_task_dead: task %s has no worker pid yet", task_id)
                    return False
            except Exception:
                # One unusable cache entry must not hide the task behind it.
                logger.debug("ensure_task_dead: skipping unreadable cache entry", exc_info=True)
                continue

            # Last look before we signal: see the race note in the docstring.
            if result.ready():
                logger.debug("ensure_task_dead: task %s finished before the kill", task_id)
                return False

            state.consumer.pool.terminate_job(worker_pid, signal.SIGKILL)
            logger.warning(
                "Task %s survived SIGTERM and was still running after the grace period;"
                " sent SIGKILL to worker process %s.",
                task_id,
                worker_pid,
            )
            return True

        logger.debug("ensure_task_dead: task %s is no longer in the pool cache", task_id)
        return False
    except Exception:
        logger.warning("ensure_task_dead: kill check failed for task %s", task_id, exc_info=True)
        return False


@control_command(args=[("task_id", str), ("grace_seconds", float)])
def ensure_task_dead(
    state,
    task_id: str,
    grace_seconds: float = TERMINATE_GRACE_SECONDS,
    timer_factory=threading.Timer,
    **_kwargs,
):
    """Schedule a SIGKILL for ``task_id`` if it is still running after the grace period.

    The handler runs in the worker main process and must return at once, so the
    check is deferred to a daemon timer thread rather than waited on here.

    Only the worker that actually holds the task schedules anything. The
    broadcast has no destination and reaches the whole fleet, so a backfill
    canceling hundreds of runs would otherwise park one timer thread per run in
    every worker main process at once. Non-owners return immediately; the owner
    re-walks the cache when the timer fires, because the task can still exit
    on its own inside the grace window.

    A worker running an older build simply does not have this command and
    ignores the broadcast — a silent no-op by design, so the fix can roll out
    without a lockstep restart of launcher and workers.
    """
    logger = logging.getLogger(__name__)
    try:
        grace = _coerce_grace_seconds(grace_seconds)

        if _find_live_result(state, task_id) is None:
            logger.debug("ensure_task_dead: task %s is not held by this worker", task_id)
            return {"ok": f"task {task_id} not held by this worker"}

        if not grace:
            _ensure_task_dead_now(state, task_id)
            return {"ok": f"checked {task_id} immediately"}

        timer = timer_factory(grace, lambda: _ensure_task_dead_now(state, task_id))
        timer.daemon = True
        timer.start()
        logger.debug(
            "ensure_task_dead: scheduled SIGKILL check for task %s in %ss",
            task_id,
            grace,
        )
        return {"ok": f"scheduled SIGKILL check for {task_id} in {grace}s"}
    except Exception as exc:
        # A control handler must never crash the worker's consumer.
        logger.warning("ensure_task_dead: could not schedule kill check for %s", task_id)
        return {"error": str(exc)}
