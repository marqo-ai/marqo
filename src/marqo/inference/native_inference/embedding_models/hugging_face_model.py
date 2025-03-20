import os
import tarfile
import zipfile
from typing import Tuple, Callable, Optional

import torch
import torch.nn.functional as F
from numpy import ndarray
from pydantic import ValidationError
from torch import Tensor
from transformers import (AutoModel, AutoTokenizer)

from marqo import marqo_docs
from marqo.core.exceptions import InternalError
from marqo.core.inference.api.modality import Modality
from marqo.inference.model_download.model_download import download_model
from marqo.inference.native_inference.embedding_models.abstract_embedding_model import AbstractEmbeddingModel
from marqo.inference.native_inference.embedding_models.abstract_preprocessor import AbstractPreprocessor
from marqo.inference.native_inference.embedding_models.hugging_face_model_properties import HuggingFaceModelProperties, \
    PoolingMethod, HuggingFaceModelFlags, HuggingFaceTokenizerFlags
from marqo.s2_inference.configs import ModelCache
from marqo.s2_inference.errors import InvalidModelPropertiesError
from marqo.s2_inference.types import List
from marqo.tensor_search.models.private_models import ModelAuth


class HuggingFacePreprocessor(AbstractPreprocessor):
    """The abstract base class for all Hugging Face preprocessors."""

    def __init__(self):
        super().__init__()

    def preprocess(self, inputs: List[str], modality: Modality) -> List[str]:
        # No preprocessing is needed for Hugging Face models
        return inputs


class HuggingFaceModel(AbstractEmbeddingModel):
    """The concrete class for all sentence transformers models loaded from Hugging Face."""

    def __init__(
            self,
            model_properties: dict,
            device: str,
            model_auth: Optional[ModelAuth] = None,
            model_flags: Optional[HuggingFaceModelFlags] = None,
            tokenizer_flags: Optional[HuggingFaceTokenizerFlags] = None
    ):
        super().__init__(model_properties=model_properties, device=device, model_auth=model_auth)

        self._model_flags = model_flags or HuggingFaceModelFlags()
        self._tokenizer_flags = tokenizer_flags or HuggingFaceTokenizerFlags()

        self.model_properties = self._build_model_properties(model_properties)

        self._model = None
        self._tokenizer = None
        self._pooling_func = None
        self._preprocessor = HuggingFacePreprocessor()

    def _build_model_properties(self, model_properties: dict) -> HuggingFaceModelProperties:
        """Convert the user input model_properties to HuggingFaceModelProperties."""
        try:
            parsed_properties = HuggingFaceModelProperties(**model_properties)
        except ValidationError as e:
            raise InvalidModelPropertiesError(f"Invalid model properties: {model_properties}. Original error {e}") \
                from e

        if self._model_flags.trust_remote_code or self._tokenizer_flags.trust_remote_code:
            if not parsed_properties.trust_remote_code:
                raise InvalidModelPropertiesError(
                    f"The specified model requires the 'trustRemoteCode' attribute to be set to True. "
                    f"Setting this attribute to True may have security implications. "
                    f"See {marqo_docs.hugging_face_trust_remote_code()} for more information"
                )

        return parsed_properties

    def _check_loaded_components(self):
        if self._model is None:
            raise InternalError("Model is not loaded!")
        if self._tokenizer is None:
            raise InternalError("Tokenizer is not loaded!")
        if self._pooling_func is None:
            raise InternalError("Pooling function is not loaded!")

    def _load_necessary_components(self):
        """Load the necessary components for the hf model.

        Raises:
            InvalidModelPropertiesError: If the model properties are invalid or incomplete.
        """

        if self.model_properties.name:
            self._model, self._tokenizer = self._load_from_hugging_face_repo()
        elif self.model_properties.url or (
                self.model_properties.model_location and self.model_properties.model_location.s3):
            self._model, self._tokenizer = self._load_from_zip_file()
        elif self.model_properties.model_location and self.model_properties.model_location.hf:
            if self.model_properties.model_location.hf.filename:
                self._model, self._tokenizer = self._load_from_zip_file()
            else:
                self._model, self._tokenizer = self._load_from_private_hugging_face_repo()
        else:
            raise InternalError(f"Invalid model properties: {self.model_properties}. Marqo can not load the model via "
                                f"a specified method. Please check the model properties and try again.")

        self._model = self._model.to(self.device)
        self._pooling_func = self._load_pooling_method()
        self._model.eval()

    def _load_from_private_hugging_face_repo(self) -> Tuple:
        """Load the model from the private Hugging Face model hub based on the model_location."""

        hf_repo_token = None
        if self.model_auth is not None and self.model_auth.hf is not None:
            hf_repo_token = self.model_auth.hf.token

        try:
            model = AutoModel.from_pretrained(
                self.model_properties.model_location.hf.repo_id,
                token=hf_repo_token,
                **self._model_flags.dict(exclude_none=True)
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_properties.model_location.hf.repo_id,
                token=hf_repo_token,
                **self._tokenizer_flags.dict(exclude_none=True)
            )
        except (OSError, ValueError, RuntimeError) as e:
            raise InvalidModelPropertiesError(
                f"Marqo encountered an error loading the private Hugging Face model, modelProperties={self.model_properties}. "
                f"Please ensure that the model is a valid Hugging Face model and retry.\n"
                f" Original error message = {e}") from e
        return model, tokenizer

    def _load_from_hugging_face_repo(self) -> Tuple:
        """Load the model from the Hugging Face model hub based on the repo_id."""
        try:
            model = AutoModel.from_pretrained(
                self.model_properties.name, **self._model_flags.dict(exclude_none=True)
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_properties.name, **self._tokenizer_flags.dict(exclude_none=True)
            )
        except (OSError, ValueError, RuntimeError) as e:
            raise InvalidModelPropertiesError(
                f"Marqo encountered an error loading the Hugging Face model, modelProperties={self.model_properties}. "
                f"Please ensure that the model is a valid Hugging Face model and retry.\n"
                f" Original error message = {e}") from e
        return model, tokenizer

    def _load_from_zip_file(self) -> Tuple:
        """Load the model from a zip file."""
        zip_file_path = download_model(
            repo_location=self.model_properties.model_location,
            url=self.model_properties.url,
            auth=self.model_auth,
            download_dir=ModelCache.hf_cache_path
        )

        model_dir = self.extract_huggingface_archive(zip_file_path)
        try:
            model = AutoModel.from_pretrained(
                model_dir,
                **self._model_flags.dict(exclude_none=True)
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir, **self._tokenizer_flags.dict(exclude_none=True)
            )
        except (OSError, ValueError, RuntimeError) as e:
            raise InvalidModelPropertiesError(
                f"Marqo encountered an error loading the Hugging Face model, modelProperties={self.model_properties}. "
                f"Please ensure that the provided zip file is valid.\n"
                f" Original error message = {e}") from e
        return model, tokenizer

    def _load_pooling_method(self) -> Callable:
        """Load the pooling method for the model."""
        if self.model_properties.pooling_method == PoolingMethod.Mean:
            return self._average_pool_func
        elif self.model_properties.pooling_method == PoolingMethod.CLS:
            return self._cls_pool_func
        else:
            raise InternalError(f"Invalid pooling method: {self.model_properties.pooling_method}")

    def encode(self, inputs: List[str], modality, normalize=True) -> List[ndarray]:

        if not isinstance(inputs, list) or not isinstance(inputs[0], str):
            raise ValueError(f"The input data should be a list of strings, but received: {inputs}")

        if self._model is None:
            self.load()

        tokenized_texts = self._tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.model_properties.tokens,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_output = self._model(**tokenized_texts)

        attention_mask = tokenized_texts['attention_mask']

        embeddings = self._pooling_func(model_output, attention_mask)

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return self._convert_output(embeddings)

    def _convert_output(self, output: Tensor) -> List[ndarray]:
        if self.device == 'cpu':
            return [single_ndarray for single_ndarray in output.numpy()]
        elif self.device.startswith('cuda'):
            return [single_ndarray for single_ndarray in output.cpu().numpy()]

    @staticmethod
    def _average_pool_func(model_output, attention_mask):
        """A pooling function that averages the hidden states of the model."""
        last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @staticmethod
    def _cls_pool_func(model_output, attention_mask=None):
        """A pooling function that extracts the CLS token from the model."""
        return model_output[0][:, 0]

    @staticmethod
    def extract_huggingface_archive(path: str) -> str:
        '''
            This function takes the path as input. The path is a string that can be one of the following:
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
            # return the directory path or repo_id directory
            return path

    def get_preprocessor(self):
        return self._preprocessor