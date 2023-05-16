from marqo.tensor_search.models.external_apis.hf import HfAuth, HfModelLocation
from typing import Optional
from huggingface_hub import hf_hub_download
from marqo.s2_inference.logger import get_logger
from huggingface_hub.utils._errors import RepositoryNotFoundError
from marqo.s2_inference.errors import ModelDownloadError

logger = get_logger(__name__)


def download_model_from_hf(
        location: HfModelLocation,
        auth: Optional[HfAuth] = None,
        download_dir: Optional[str] = None):
    """Downloads a pretrained model from HF, if it doesn't exist locally. The basename of the
    location's filename is used as the local filename.

    hf_hub_download downloads the model if it does not yet exist in the cache.

    Args:
        location: repo_id and filename to be downloaded.
        auth: contains HF API token for model access
        download_dir: The location where the model
            should be stored

    Returns:
        Path to the downloaded model
    """
    download_kwargs = location.dict(exclude_unset=True) # Ignore unset values to avoid adding None to params
    if auth is not None:
        download_kwargs = {**download_kwargs, **auth.dict()}
    try:
        return hf_hub_download(**download_kwargs, cache_dir=download_dir)
    except RepositoryNotFoundError:
        # TODO: add link to HF model auth/loc
        raise ModelDownloadError(
            "Could not find the specified Hugging Face model repository. Please ensure that the request's model_auth's "
            "`hf` credentials and the index's model_location are correct. "
            "If the index's model_location is not correct, please create a new index with the corrected model_location"
        )

