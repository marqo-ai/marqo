import os, validators
import zipfile, tarfile
from urllib.error import HTTPError
import numpy as np
from typing import Optional
import torch
from torch import nn
from transformers import (AutoModel, AutoTokenizer)
from marqo.tensor_search.models.private_models import ModelLocation, ModelAuth
from marqo.tensor_search.enums import ModelProperties, InferenceParams
from marqo.s2_inference.sbert_utils import Model
from marqo.s2_inference.types import Union, FloatTensor, List
from marqo.s2_inference.logger import get_logger
from marqo.tensor_search.enums import ModelProperties
from marqo.s2_inference.errors import InvalidModelPropertiesError, ModelDownloadError
from marqo.s2_inference.processing.custom_clip_utils import download_model
from marqo.s2_inference.configs import ModelCache



logger = get_logger(__name__)


class HF_MODEL(Model):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        if self.max_seq_length is None:
            self.max_seq_length = 128
        self.model_properties = kwargs.get("model_properties", dict())
        self.model_name = self.model_properties.get("name", None)
        self.model_auth = kwargs.get(InferenceParams.model_auth, None)

    def load(self) -> None:

        model_location_presence = ModelProperties.model_location in self.model_properties
        path = self.model_properties.get("localpath", None) or self.model_properties.get("url", None)
        # HF models can be loaded from 3 entries: path (url or localpath), model_name, or model_location
        if (path is not None) + (self.model_name is not None) + (model_location_presence is True) != 1:
            raise InvalidModelPropertiesError("Exactly one of (`localpath`/`url`) or `model_location`, `name` can be specified"
                                              " in `model_properties` for `hf` models as they conflict with each other in model loading."
                                              " Please ensure that exactly one of these is specified in `model_properties` and retry.")
        elif path is not None:
            if validators.url(path) is True:
                self.model_path = download_model(url = path, download_dir=ModelCache.hf_cache_path)
            elif os.path.isdir(path) or os.path.isfile(path):
                self.model_path = path
        elif self.model_name is not None:
            # Loading from structured huggingface repo directly, token is required directly
            self.model_path = self.model_name
        elif model_location_presence is True:
            # This is a special case for huggingface models, where we can load a model directory from a repo
            if ("hf" in self.model_properties["model_location"]) and ("repo_id" in self.model_properties["model_location"]["hf"]) and \
                ("filename" not in self.model_properties["model_location"]["hf"]):
                return self._load_from_private_hf_repo()
            else:
                self.model_path = self._download_from_repo()

        # We need to do extraction here if necessary
        self.model_path = extract_huggingface_archive(self.model_path)

        self.model = AutoModelForSentenceEmbedding(self.model_path).to(self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except (OSError, ValueError, RuntimeError) as e:
            raise InvalidModelPropertiesError(
                f"Marqo encountered an error loading the Hugging Face model = `{self.model_path}` using AutoTokenizer "
                f"Please ensure that the model is a valid Hugging Face model and retry.\n"
                f" Original error message = {e}")
        except (HTTPError, ConnectionError) as e:
            raise ModelDownloadError(
                f"Marqo encountered an ConnectionError loading the Hugging Face model = `{self.model_path}` using AutoTokenizer. "
                f"This is likely to be caused by an internet issue. Please check Marqo's internet connection to Hugging Face and retry. \n"
                f" Original error message = {e}")

    def _load_from_private_hf_repo(self) -> None:
        """
        Load a private model from a huggingface repo directly using the `repo_id` attribute in `model_properties`
        This is a special case for HF models, where we can load a model directory from a repo.
        The self.model_path will be set to the repo_id, which is the remote path in the HuggingFace repo.
        Token is also used if provided in `model_auth` object.
        """
        model_location = ModelLocation(**self.model_properties[ModelProperties.model_location])
        self.model_path = model_location.hf.repo_id

        token = None
        if model_location.auth_required:
            try:
                token = self.model_auth.hf.token
            except AttributeError:
                raise InvalidModelPropertiesError("Please ensure that `model_auth` is valid for a private Hugging Face model and retry. "
                                                  "A valid `ModelAuth` object should consist a `hugging face token` attribute for private hf repo models")

        self.model = AutoModelForSentenceEmbedding(model_name=self.model_path, use_auth_token=token).to(self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_auth_token=token)
        except (OSError, ValueError, RuntimeError) as e:
            raise InvalidModelPropertiesError(f"Marqo encounterend an error loading the Hugging Face model = `{self.model_path}` using AutoTokenizer "
                                              f"Please ensure that the model is a valid Hugging Face model, the token is correct, and retry\n"
                                              f" Original error message = {e}")
        except (HTTPError, ConnectionError) as e:
            raise ModelDownloadError(f"Marqo encounters ConnectionError loading the Hugging Face model = `{self.model_path}` using AutoTokenizer. "
                                     f"This is likely to be caused by an internet issue. Please check Marqo's internet connection to Hugging Face and retry. \n"
                                     f" Original error message = {e}")

    def _download_from_repo(self) -> str:
        """Downloads model from an external repo like s3 and returns the filepath

        Returns:
            The model's filepath or a string of hugging face repo name

        Raises:
            RunTimeError if an empty filepath is detected.
        """
        model_location = ModelLocation(**self.model_properties[ModelProperties.model_location])
        download_model_params = {"repo_location": model_location}

        if model_location.auth_required:
            download_model_params['auth'] = self.model_auth

        model_file_path = download_model(**download_model_params, download_dir=ModelCache.hf_cache_path)
        if model_file_path is None or model_file_path == '':
            raise RuntimeError(
                'download_model() needs to return a valid filepath to the model! Instead, received '
                f' filepath `{model_file_path}`')
        return model_file_path

    def encode(self, sentence: Union[str, List[str]], normalize=True, **kwargs) -> Union[FloatTensor, np.ndarray]:

        if isinstance(sentence, str):
            sentence = [sentence]

        if self.model is None:
            self.load()

        self.model.normalize = normalize
        inputs = self.tokenizer(sentence, padding=True, truncation=True, max_length=self.max_seq_length,
                                return_tensors="pt").to(self.device)

        with torch.no_grad():
            return self._convert_output(self.model.forward(**inputs))

    def _convert_output(self, output):
        if self.device == 'cpu':
            return output.numpy()
        elif self.device.startswith('cuda'):
            return output.cpu().numpy()


class AutoModelForSentenceEmbedding(nn.Module):

    def __init__(self, model_name: Optional[str] = None, use_auth_token: Optional[str] = None, normalize=True, pooling='mean'):
        super().__init__()
        self.model_name = model_name
        self.normalize = normalize
        self.pooling = pooling
        try:
            self.model = AutoModel.from_pretrained(model_name, use_auth_token = use_auth_token, cache_dir=ModelCache.hf_cache_path)
        except (OSError, ValueError, RuntimeError) as e:
            raise InvalidModelPropertiesError(
                f"Marqo encounters error loading the Hugging Face model = `{self.model_path}` using AutoModel "
                f"Please ensure that the model is a valid Hugging Face model and retry.\n"
                f" Original error message = {e}")
        except (HTTPError, ConnectionError) as e:
            raise ModelDownloadError(
                f"Marqo encounters ConnectionError loading the Hugging Face model = `{self.model_path}` using AutoModel. "
                f"This is likely to be caused by an internet issue. Please check Marqo's internet connection with Hugging Face and retry. \n"
                f" Original error message = {e}")
        self.model.eval()
        if self.pooling == 'mean':
            self._pool_func = self.mean_pooling
        elif self.pooling == 'cls':
            self._pool_func = self.cls_pooling
        else:
            raise TypeError(f"{pooling} not in allowed pooling types of 'mean' or 'cls' ")

    def forward(self, **kwargs):

        model_output = self.model(**kwargs)

        embeddings = self._pool_func(model_output, kwargs['attention_mask'])

        if self.normalize:
            return nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def mean_pooling(self, model_output, attention_mask):

        token_embeddings = model_output[0]

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def cls_pooling(self, model_output, attention_mask):
        return model_output[0][:, 0]


def extract_huggingface_archive(path: str) -> str:
    '''

        This function takes the path as input. The path can must be a string that can be:
        1. A downloaded archive file. This function will extract the model from the archive return the directory path.
        2. A repo_id in huggingface. This function will return the input string directly.

        path: the downloaded model archive path or a repo_id in huggingface
    Returns:
        The directory path to the model or the repo_id in huggingface
    '''
    if os.path.isfile(path):
        # if it's a file, check if it's a compressed file
        base, ext = os.path.splitext(path)
        if ext in ['.bin', '.pt']:
            raise InvalidModelPropertiesError(f"Marqo does not support loading Hugging Face SBERT models from the provided single `{ext}` file. "
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
            raise InvalidModelPropertiesError(f'Marqo encountered an error while extracting the compressed model archive from `{path}`.\n '
                                              f'This is probably because the file is corrupted or the extension `{ext}` is not supported. '
                                              f'Marqo has removed the corrupted file from the disk.'
                                              f'Please ensure that the file is a valid compressed file and try again.')
        # will this error really happen?
        except PermissionError:
            raise InvalidModelPropertiesError(f'Marqo encountered an error while extracting the compressed model archive from `{path}`. '
                                              f'This is probably because the Marqo does not have the permission to write to the directory. '
                                              f'Please check the access permission of Marqo and try again.')
        except Exception as e:
            raise RuntimeError(f'Marqo encountered an error while extracting the compressed model archive from `{path}`. '
                                              f'The original error message is `{str(e)}`')
    else:
        # return the directory path or repo_id directory
        return path
