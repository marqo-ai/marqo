import os
import threading
from contextlib import contextmanager

from jinja2 import Environment, PackageLoader
from model_management.core.logging import get_logger
from model_management.schemas.triton_model_properties import TritonModelProperties
from model_management.services.model_manager.triton_model_downloader import (
    TritonModelDownloader,
)
from model_management.services.triton.triton_client import TritonClient

from ..errors import ModelOperationInProgressError

logger = get_logger(__name__)

env = Environment(
    loader=PackageLoader("model_management.services.model_manager", "templates")
)
template = env.get_template("config_pbtxt_template.jinja2")

_MODEL_IO_LOCK = threading.Lock()


@contextmanager
def _model_op_guard(lock: threading.Lock, timeout: float = 2.0):
    """Try to acquire the lock for model operations. Wait for up to 2 seconds to avoid
    bursts of requests causing immediate failures.

    Raise ModelOperationInProgressError if it can not acquire the lock.
    """
    acquired = lock.acquire(timeout=timeout)
    if not acquired:
        raise ModelOperationInProgressError(
            "Another model load/unload operation is in progress. Please try again later"
        )
    try:
        yield
    finally:
        lock.release()


class ModelManager:
    def __init__(self, marqo_model_cache_path: str, triton_client: TritonClient):
        self.marqo_model_cache_path = marqo_model_cache_path
        self.triton_client = triton_client

    def load_model(self, triton_model_properties: TritonModelProperties) -> None:
        with _model_op_guard(_MODEL_IO_LOCK):
            logger.info(f"Loading model: {triton_model_properties.model_dump_json()}")
            TritonModelDownloader(
                sources=triton_model_properties.sources,
                base_dir=self.marqo_model_cache_path,
                model_name=triton_model_properties.name,
                config_pbtxt=self.generate_config_pbtxt_file(triton_model_properties),
                overwrite=False,
            ).prepare_and_download()

            self.triton_client.load_model(triton_model_properties.name)
            logger.info(f"Model loaded: {triton_model_properties.name}")

    def unload_model(self, model_name: str, remove_files: bool = False) -> None:
        with _model_op_guard(_MODEL_IO_LOCK):
            logger.info(f"Unloading model: {model_name}")
            self.triton_client.unload_model(model_name)

            if remove_files:
                model_dir = os.path.join(self.marqo_model_cache_path, model_name)
                if os.path.exists(model_dir):
                    for root, dirs, files in os.walk(model_dir, topdown=False):
                        for name in files:
                            os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                    os.rmdir(model_dir)
                    logger.info(f"Removed model files for: {model_name}")
            logger.info(f"Model unloaded: {model_name}")

    @staticmethod
    def generate_config_pbtxt_file(
        triton_model_properties: TritonModelProperties,
    ) -> str:
        # Implement the logic to generate a config.pbtxt file for Triton Inference Server
        context = {
            "name": triton_model_properties.name,
            "max_batch_size": triton_model_properties.max_batch_size,
            "input": triton_model_properties.input,
            "output": triton_model_properties.output,
        }

        rendered = template.render(**context)
        return rendered
