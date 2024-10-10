import os
import tarfile
import zipfile
from typing import Tuple, Callable

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import ValidationError
from sympy.codegen.fnodes import dimension
from transformers import (AutoModel, AutoTokenizer)

from marqo import marqo_docs
from marqo.core.inference.model_download import download_model
from marqo.core.inference.inference_models.abstract_embedding_model import AbstractEmbeddingModel
from marqo.core.inference.inference_models.hugging_face_model_properties import HuggingFaceModelProperties, \
    PoolingMethod
from marqo.s2_inference.configs import ModelCache
from marqo.s2_inference.errors import InvalidModelPropertiesError
from marqo.s2_inference.types import Union, FloatTensor, List


def _average_pool_func(model_output, attention_mask):
    last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def _cls_pool_func(model_output, attention_mask=None):
    return model_output[0][:, 0]


class HuggingFaceStellaModel(AbstractEmbeddingModel):
    """The concrete class for all sentence transformers models loaded from Hugging Face."""

    def __init__(self, model_properties: dict, device: str, model_auth: dict):
        super().__init__(model_properties, device, model_auth)

        self.model_properties = self._build_model_properties(model_properties)

        self.model = None
        self.tokenizer = None
        self.vector_linear = None
        self.pooling_func = None

    def _build_model_properties(self, model_properties: dict) -> HuggingFaceModelProperties:
        try:
            return HuggingFaceModelProperties(**model_properties)
        except ValidationError as e:
            raise InvalidModelPropertiesError(f"Invalid model properties for the 'hf' model. Original error {e}") \
                from e

    def _check_loaded_components(self):
        if self.model is None:
            raise RuntimeError("Model is not loaded!")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded!")
        if self.pooling_func is None:
            raise RuntimeError("Pooling function is not loaded!")

    def _load_necessary_components(self):
        if self.model_properties.name:
            self.model, self.tokenizer = self._load_from_hugging_face_repo()
        elif self.model_properties.url:
            self.model, self.tokenizer, self.vector_linear = self._load_from_zip_file()
        elif self.model_properties.model_location:
            if self.model_properties.model_location.s3:
                self.model, self.tokenizer = self._load_from_zip_file()
            elif self.model_properties.model_location.hf:
                if self.model_properties.model_location.hf.filename:
                    self.model, self.tokenizer = self._load_from_zip_file()
                else:
                    self.model, self.tokenizer = self._load_from_private_hugging_face_repo()
            else:
                raise InvalidModelPropertiesError(
                    f"Invalid model properties for the 'hf' model. "
                    f"You do not have the necessary information to load the model. "
                    f"Check {marqo_docs.bring_your_own_model()} for more information."
                )
        else:
            raise InvalidModelPropertiesError(
                f"Invalid model properties for the 'hf' model. "
                f"You do not have the necessary information to load the model. "
                f"Check {marqo_docs.bring_your_own_model()} for more information."
            )
        self.model = self.model.to(self.device)
        self.pooling_func = self._load_pooling_method()
        self.model.eval()

    def _load_from_private_hugging_face_repo(self) -> Tuple:
        """Load the model from the private Hugging Face model hub based on the model_location."""

        hf_repo_token = None
        if self.model_auth is not None and self.model_auth.hf is not None:
            hf_repo_token = self.model_auth.hf.token

        try:
            model = AutoModel.from_pretrained(
                self.model_properties.model_location.hf.repo_id,
                token=hf_repo_token
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_properties.model_location.hf.repo_id,
                token=hf_repo_token
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
            model = AutoModel.from_pretrained(self.model_properties.name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(self.model_properties.name, trust_remote_code=True)
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

        dimensions = self.model_properties.dimensions
        model_dir = extract_huggingface_archive(zip_file_path)
        vector_linear_directory = f"2_Dense_{dimensions}"
        try:
            model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,
                                              use_memory_efficient_attention=False,
                                              unpad_inputs=False).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            vector_linear = torch.nn.Linear(in_features=model.config.hidden_size,
                                            out_features=dimensions)
            vector_linear_dict = {
                k.replace("linear.", ""): v for k, v in
                torch.load(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin"),
                           map_location=torch.device(self.device)).items()
            }
            vector_linear.load_state_dict(vector_linear_dict)
            vector_linear.to(self.device)
        except (OSError, ValueError, RuntimeError) as e:
            raise InvalidModelPropertiesError(
                f"Marqo encountered an error loading the Hugging Face model, modelProperties={self.model_properties}. "
                f"Please ensure that the provided zip file is valid.\n"
                f" Original error message = {e}") from e
        return model, tokenizer, vector_linear

    def _load_pooling_method(self) -> Callable:
        """Load the pooling method for the model."""
        if self.model_properties.pooling_method == PoolingMethod.Mean:
            return _average_pool_func
        elif self.model_properties.pooling_method == PoolingMethod.CLS:
            return _cls_pool_func
        else:
            raise ValueError(f"Invalid pooling method: {self.model_properties.pooling_method}")

    def encode(self, sentence: Union[str, List[str]], normalize=True, **kwargs) -> Union[FloatTensor, np.ndarray]:
        if isinstance(sentence, str):
            sentence = [sentence]

        if self.model is None:
            self.load()

        self.model.normalize = normalize
        tokenized_texts = self.tokenizer(
            sentence,
            padding=True,
            truncation=True,
            max_length=self.model_properties.token,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**tokenized_texts)
            attention_mask = tokenized_texts['attention_mask']
            embeddings = self.vector_linear(self.pooling_func(model_output, attention_mask))

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return self._convert_output(embeddings)

    def _convert_output(self, output):
        if self.device == 'cpu':
            return output.numpy()
        elif self.device.startswith('cuda'):
            return output.cpu().numpy()


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
