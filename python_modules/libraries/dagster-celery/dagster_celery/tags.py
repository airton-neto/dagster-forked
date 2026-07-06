# Used to set the priority for a particular step execution
DAGSTER_CELERY_STEP_PRIORITY_TAG = "dagster-celery/priority"

# Used to set the priority for an overall job run
DAGSTER_CELERY_RUN_PRIORITY_TAG = "dagster-celery/run_priority"

# Used to select a Celery queue
DAGSTER_CELERY_QUEUE_TAG = "dagster-celery/queue"

# Used to set the celery task_id for run monitoring
DAGSTER_CELERY_TASK_ID_TAG = "dagster-celery/task_id"

# Written by the run worker task when it starts executing a run. Run monitoring
# pings this hostname; unlike the result-backend task meta, it cannot be
# overwritten by a duplicate (redelivered) task execution on another worker.
DAGSTER_CELERY_WORKER_HOSTNAME_TAG = "dagster-celery/worker-hostname"
