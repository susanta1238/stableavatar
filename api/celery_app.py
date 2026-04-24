from celery import Celery
from api.config import REDIS_URL

celery_app = Celery(
    "stableavatar",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    broker_connection_retry_on_startup=True,
)

import api.tasks  # noqa: E402,F401  register tasks
