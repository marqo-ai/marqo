import os

from inference_orchestrator.core.logging import get_logger
from inference_orchestrator.core.settings import get_settings

settings = get_settings()

logger = get_logger(__name__)


class ModelDownloadCache:
    """
    A class to manage cache paths for different model types.
    """

    open_clip_cache_path = os.path.join(
        settings.marqo_model_cache_path, "marqo-open-clip-cache/"
    )
    # The hf_cache_path is managed by the hf_hub_download function
    hf_cache_path = os.path.join(settings.marqo_model_cache_path, "marqo-hf-cache/")
