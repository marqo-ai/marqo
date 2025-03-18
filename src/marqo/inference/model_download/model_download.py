import os
import tarfile
import urllib
import zipfile
from typing import Union, Optional
from urllib.error import HTTPError

from tqdm import tqdm

from marqo.core.exceptions import InternalError
from marqo.inference.model_download.download_model_from_hf import download_model_from_hf
from marqo.inference.model_download.download_model_from_s3 import (
    get_presigned_s3_url, get_s3_model_cache_filename, check_s3_model_already_exists,
    get_s3_model_absolute_cache_path
)
from marqo.s2_inference.configs import ModelCache
from marqo.s2_inference.errors import ModelDownloadError, InvalidModelPropertiesError
from marqo.tensor_search.models.external_apis.s3 import S3Auth, S3Location
from marqo.tensor_search.models.private_models import ModelAuth, ModelLocation


def download_model(
        repo_location: Optional[ModelLocation] = None,
        url: Optional[str] = None,
        auth: Optional[ModelAuth] = None,
        download_dir: Optional[str] = None
    ) -> str:
    """
    Download a model from a given location.

    Args:
        repo_location: object that contains information about the location of a
            model. For example, s3 bucket and object path
        url: location of a model specified by a URL
        auth: object that contains information about authorisation required to
            download a model. For example, s3 access keys
        download_dir: The directory where the model should be downloaded.

    Returns:
        The path of the downloaded model
    """
    single_weight_location_validation_msg = (
        "only exactly one of parameters (repo_location, url) is allowed to be specified.")
    if repo_location is None and url is None:
        raise InvalidModelPropertiesError(single_weight_location_validation_msg)
    if repo_location is not None and url is not None:
        raise InvalidModelPropertiesError(single_weight_location_validation_msg)

    if url:
        return download_pretrained_from_url(url=url, cache_dir=download_dir)
    if isinstance(repo_location, ModelLocation):
        if repo_location.s3:
            download_kwargs = {'location': repo_location.s3, 'download_dir': download_dir}
            if auth is not None:
                download_kwargs['auth'] = auth.s3
            return download_pretrained_from_s3(**download_kwargs)
        elif repo_location.hf:
            download_kwargs = {'location': repo_location.hf, 'download_dir': download_dir}
            if auth is not None:
                download_kwargs['auth'] = auth.hf
            return download_model_from_hf(**download_kwargs)
    else:
        raise InternalError("Invalid model location object provided.")


def download_pretrained_from_s3(
        location: S3Location,
        auth: Optional[S3Auth] = None,
        download_dir: Optional[str] = None
) -> str:
    """Downloads a pretrained model from S3, if it doesn't exist locally. The basename of the object's
    key is used for the filename.

    Args:
        location: Bucket and key of model file to be downloaded
        auth: AWS IAM access keys to a user with access to the model to be downloaded
        download_dir: the location where the model should be stored

    Returns:
        Path to the downloaded model
    """
    if check_s3_model_already_exists(location=location, download_dir=download_dir):
        # TODO: check if abs path is even the most appropriate???
        return get_s3_model_absolute_cache_path(location=location, download_dir=download_dir)

    url = get_presigned_s3_url(location=location, auth=auth)

    try:
        return download_pretrained_from_url(
            url=url, cache_dir=download_dir,
            cache_file_name=get_s3_model_cache_filename(location)
        )
    except HTTPError as e:
        if e.code == 403:
            # TODO: add link to auth docs
            raise ModelDownloadError(
                "Received 403 error when trying to retrieve model from s3 storage. "
                "Please check the request's s3 credentials and try again. "
            ) from e
        else:
            raise e

def download_pretrained_from_url(
        url: str,
        cache_dir: Union[str, None] = None,
        cache_file_name: Optional[str] = None,
) -> str:
    '''
    This function takes a clip model checkpoint url as input, downloads the model if it doesn't exist locally,
    and returns the local path of the downloaded file.

    Args:
        url: a valid string of the url address.
        cache_dir: the directory to store the file
        cache_file_name: name of the model file when it gets downloaded to the cache.
            If not provided, the basename of the URL is used.
    Returns:
        download_target: the local path of the downloaded file.
    '''
    buffer_size = 8192
    if not cache_dir:
        cache_dir = os.path.expanduser(ModelCache.clip_cache_path)
    os.makedirs(cache_dir, exist_ok=True)

    if cache_file_name is None:
        filename = os.path.basename(url)
    else:
        filename = cache_file_name

    download_target = os.path.join(cache_dir, filename)

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.headers.get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(buffer_size)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target


def extract_zip_file(path: str) -> str:
    """
    This function extracts a compressed file to a new directory with the same name as the file. The directory is
    normally used to load the model.
    """
    if os.path.isfile(path):
        # if it's a file, check if it's a compressed file
        base, ext = os.path.splitext(path)
        if ext in ['.bin', '.pt']:
            raise InvalidModelPropertiesError(
                f"Marqo does not support loading Hugging Face SBERT models from the provided single `{ext}` file. "
                "Please try to wrap the model in a Hugging Face archive file and try again. ")

        try:
            # create a new directory with the same name as the file
            new_dir = base
            os.makedirs(new_dir, exist_ok=True)

            # extract the compressed file
            # If the target directory already exists, it will be overwritten by default without warning.
            if ext == '.zip':
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(new_dir)
            else:
                with tarfile.open(path, 'r') as tar_ref:
                    tar_ref.extractall(new_dir)
            # return the path to the new directory
            return new_dir
        except (tarfile.ReadError, zipfile.BadZipfile):
            try:
                os.remove(path)
            except Exception as remove_e:
                raise RuntimeError(
                    f"Marqo encountered an error while attempting to delete a corrupted file `{path}`. "
                    f"Please report this issue on Marqo's Github Repo and replace the problematic Marqo instance with "
                    f"a new one. \n "
                    f"Error message: `{str(remove_e)}`"
                )
            raise InvalidModelPropertiesError(
                f'Marqo encountered an error while extracting the compressed model archive from `{path}`.\n '
                f'This is probably because the file is corrupted or the extension `{ext}` is not supported. '
                f'Marqo has removed the corrupted file from the disk.'
                f'Please ensure that the file is a valid compressed file and try again.')

        # will this error really happen?
        except PermissionError:
            raise InvalidModelPropertiesError(
                f'Marqo encountered an error while extracting the compressed model archive from `{path}`. '
                f'This is probably because the Marqo does not have the permission to write to the directory. '
                f'Please check the access permission of Marqo and try again.')

        except Exception as e:
            raise RuntimeError(
                f'Marqo encountered an error while extracting the compressed model archive from `{path}`. '
                f'The original error message is `{str(e)}`')

    else:
        raise InternalError(
            f"The provided path does not exist. {path}"
        )

