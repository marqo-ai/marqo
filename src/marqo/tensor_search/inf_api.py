from fastapi import FastAPI
import os

from marqo.logging import get_logger
from marqo.tensor_search.main import get_config
from marqo.tensor_search.on_start_script import on_start

logger = get_logger(__name__)


logger.info(f'{os.getpid()}: {__name__} on_start')
on_start(get_config(), 'inference')

inf_app = FastAPI(
    title="Marqo Inference"
)

# TODO expose inference endpoint
# TODO replace any s2_inference.vectorise call with inference endpoint
