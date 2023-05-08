import os
from marqo.s2_inference.configs import ModelCache
from marqo.tensor_search.models.private_models import ModelAuth, ModelLocation
from marqo.tensor_search.models.external_apis.s3 import S3Auth, S3Location
from typing import Optional
from marqo.tensor_search.enums import ObjectStores
import boto3
from marqo.s2_inference.errors import ModelDownloadError
from botocore.exceptions import NoCredentialsError


def get_presigned_s3_url(location: S3Location, auth: Optional[S3Auth] = None):
    """Returns the s3 url of a request to get an S3 object

    Args:
        location: Bucket and key of model file to be downloaded
        auth: AWS IAM access keys to a user with access to the model to be downloaded

    Returns:
        The the presigned s3 URL

    TODO: add link to proper usage in error messages
    """
    if S3Auth is None:
        raise ModelDownloadError(
            "Error retrieving private model. s3 authorisation information is required to "
            "download a model from an s3 bucket. "
            "If the model is publicly accessible, please use the model's publicly accessible URL."
        )
    s3_client = boto3.client('s3', **auth.dict())
    try:
        return s3_client.generate_presigned_url('get_object', Params=location.dict())
    except NoCredentialsError:  # TODO: invalid creds error
        raise ModelDownloadError(
            "Error retrieving private model. AWS credentials were not accepted."
        )


def get_s3_model_absolute_cache_path(location: S3Location) -> str:
    """Returns the absolute path of an s3 model if it were downloaded.

        Args:
            location: Bucket and key of model file to be downloaded

        Returns:
            The absolute path of an s3 model if it were downloaded.
    """
    cache_dir = os.path.expanduser(ModelCache.clip_cache_path)
    return os.path.join(cache_dir, get_s3_model_cache_filename(location))


def check_s3_model_already_exists(location: S3Location) -> bool:
    """Returns True iff an s3 model is already downloaded

        Args:
            location: Bucket and key of model file to be downloaded

        Returns:
            The model cache filename of an s3 object
    """
    abs_path = get_s3_model_absolute_cache_path(location)
    # TODO: is isfile() the best function to check this??
    return os.path.isfile(abs_path)


def get_s3_model_cache_filename(location: S3Location) -> str:
    """Returns the model cache filename of an s3 object

    Args:
        location: Bucket and key of model file to be downloaded

    Returns:
        The model cache filename of an s3 object
    """
    return os.path.basename(location.Key)


