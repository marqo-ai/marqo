from marqo.tensor_search.models.external_apis.hf import HfAuth, HfModelLocation
from typing import Optional
from huggingface_hub import cached_download, hf_hub_url, hf_hub_download
from marqo.s2_inference.logger import get_logger


logger = get_logger(__name__)


def download_model_from_hf(
        location: HfModelLocation,
        auth: Optional[HfAuth] = None,
        download_dir: Optional[str] = None):
    """Downloads a pretrained model from HF, if it doesn't exist locally. The basename of the
    location's filename is used as the local filename.

    Args:
        location: repo_id and filename to be downloaded.
        auth: contains HF API token for model access
        download_dir: [not yet implemented]. The location where the model
            should be stored

    Returns:
        Path to the downloaded model
    """
    if download_dir is not None:
        logger.warning(
            "Hugging Face model download was given the `download_dir` argument, "
            "even though it is not yet implemented. "
            "The specified model will be downloaded but the `download_dir` "
            "parameter will be ignored."
        )
    download_kwargs = location.dict()
    if auth is not None:
        download_kwargs = {**download_kwargs, **auth.dict()}
    hf_hub_download(download_kwargs)

