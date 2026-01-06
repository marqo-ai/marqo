import os

from marqo.logging import get_logger
from marqo.s2_inference.configs import ModelCache
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.models.external_apis.s3 import S3Auth, S3Location
from typing import Optional
import boto3
from botocore.config import Config
from marqo.s2_inference.errors import ModelDownloadError
from botocore.exceptions import NoCredentialsError
from marqo.tensor_search import utils
logger = get_logger(__name__)

def get_presigned_s3_url(location: S3Location, auth: Optional[S3Auth] = None):
    """Returns the s3 url of a request to get an S3 object

    Args:
        location: Bucket and key of model file to be downloaded
        auth: AWS IAM access keys to a user with access to the model to be downloaded

    Returns:
        The presigned s3 URL

    TODO: add link to proper usage in error messages
    """
    # Require dual stack endpoint to support IPv6 addresses
    env_val = utils.read_env_vars_and_defaults(EnvVars.MARQO_MODEL_DOWNLOAD_S3_USE_DUAL_STACK) or "FALSE"
    use_dual_stack = env_val.upper() == "TRUE"
    logger.info("Using dual stack endpoint for S3: %s", use_dual_stack)

    s3_client = boto3.client('s3',config=Config(use_dualstack_endpoint=use_dual_stack), **(auth.dict() if auth is not None else {}))
    try:
        return s3_client.generate_presigned_url('get_object', Params=location.dict(exclude_unset=True))
    except NoCredentialsError:
        raise ModelDownloadError(
            "Error retrieving private model. AWS credentials were not accepted."
        )


def get_s3_model_absolute_cache_path(location: S3Location, download_dir: Optional[str] = None) -> str:
    """Returns the absolute path of an s3 model if it were downloaded.

        Args:
            location: Bucket and key of model file to be downloaded

        Returns:
            The absolute path of an s3 model if it were downloaded.
    """

    cache_dir = os.path.expanduser(download_dir if download_dir is not None else ModelCache.clip_cache_path)
    return os.path.join(cache_dir, get_s3_model_cache_filename(location))


def check_s3_model_already_exists(location: S3Location, download_dir: Optional[str] = None) -> bool:
    """Returns True iff an s3 model is already downloaded

        Args:
            location: Bucket and key of model file to be downloaded

        Returns:
            The model cache filename of an s3 object
    """
    abs_path = get_s3_model_absolute_cache_path(location, download_dir)
    return os.path.isfile(abs_path)


def get_s3_model_cache_filename(location: S3Location) -> str:
    """Returns the model cache filename of an s3 object

    Args:
        location: Bucket and key of model file to be downloaded

    Returns:
        The model cache filename of an s3 object
    """
    return os.path.basename(location.Key)


