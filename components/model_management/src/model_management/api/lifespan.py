import inspect
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..config import get_config
from ..core.logging import get_logger, instantiate_logger
from ..core.settings import get_settings
from ..on_start import on_start


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Settings
    s = get_settings()

    # 2. instantiate logger
    instantiate_logger(s)
    logger = get_logger(__name__)
    logger.info(
        "Logger configured with format=%s level=%s",
        s.marqo_log_format,
        s.marqo_log_level,
    )

    # 3. Initialize configuration
    cfg = get_config()

    # 4. Run startup tasks
    res = on_start(cfg, s)
    if inspect.isawaitable(res):
        await res

    # 5. Application runs
    yield
