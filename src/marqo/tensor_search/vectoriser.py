import io
from typing import List, Union, Protocol, Optional

import open_clip
import torch
import httpx
from orjson import orjson
import numpy as np
import os

from marqo.s2_inference import s2_inference
from marqo.s2_inference.clip_utils import _get_transform, format_and_load_CLIP_image
from marqo.s2_inference.multimodal_model_load import Modality
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.models.inf_request import VectoriseRequest, VectoriseResponse
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.telemetry import RequestMetricsStore


class InferenceError(Exception):
    pass


class Vectoriser(Protocol):
    def vectorise(self, model_name: str, content: Union[str, List[str]],
                  model_properties: dict = None, device: str = None, normalize_embeddings: bool = True,
                  model_auth: ModelAuth = None, enable_cache: bool = False, modality: Modality = Modality.TEXT,
                  media_download_headers: Optional[dict] = None, **kwargs) -> List[List[float]]:
        ...


class LocalVectoriser:
    def vectorise(self, model_name: str, content: Union[str, List[str]],
                  model_properties: dict = None, device: str = None, normalize_embeddings: bool = True,
                  model_auth: ModelAuth = None, enable_cache: bool = False, modality: Modality = Modality.TEXT,
                  media_download_headers: Optional[dict] = None, **kwargs) -> List[List[float]]:

        with RequestMetricsStore.for_request().time("inference.local_vectorise"):
            return s2_inference.vectorise(model_name=model_name, content=content, model_properties=model_properties,
                                          device=device, normalize_embeddings=normalize_embeddings, model_auth=model_auth,
                                          enable_cache=enable_cache, modality=modality,
                                          media_download_headers=media_download_headers, **kwargs)


class StaticVectoriser:
    def __init__(self, dimension: int):
        self.dimension = dimension
        random_array = np.random.rand(dimension)
        normalized_array = random_array / np.linalg.norm(random_array)
        self.static_vector = [normalized_array.tolist()]
    def vectorise(self, model_name: str, content: Union[str, List[str]],
                  model_properties: dict = None, device: str = None, normalize_embeddings: bool = True,
                  model_auth: ModelAuth = None, enable_cache: bool = False, modality: Modality = Modality.TEXT,
                  media_download_headers: Optional[dict] = None, **kwargs) -> List[List[float]]:
        return self.static_vector


class RemoteVectoriser:
    def __init__(self, inf_url: str = 'http://localhost:8881', pool_size: int = 10):
        self.inf_url = inf_url
        self.http_client = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=pool_size, max_connections=pool_size)
        )

    def vectorise(self, model_name: str, content: Union[str, List[str]],
              model_properties: dict = None, device: str = None, normalize_embeddings: bool = True,
              model_auth: ModelAuth = None, enable_cache: bool = False, modality: Modality = Modality.TEXT,
              media_download_headers: Optional[dict] = None, **kwargs) -> List[List[float]]:
        endpoint = f'{self.inf_url}/vectorise'

        request = VectoriseRequest(
            model_name=model_name,
            model_properties=model_properties,
            model_auth=model_auth,
            modality=modality,
            content=content,
            normalize_embeddings=normalize_embeddings,
            device=device,
            enable_cache=enable_cache,
            media_download_headers=media_download_headers,
            # TODO support other params
        )

        try:
            with RequestMetricsStore.for_request().time("inference.roundtrip"):
                resp = self.http_client.post(endpoint, json=request.dict())
        except httpx.HTTPError as e:
            raise InferenceError(e) from e

        resp.raise_for_status()
        result = VectoriseResponse.construct(**orjson.loads(resp.text))
        RequestMetricsStore.for_request().add_time("inference.local_vectorise", result.vectorise_time)

        return result.embeddings


class RemoteSlimVectoriser:
    def __init__(self, inf_url: str = 'http://localhost:8881', pool_size: int = 10):
        self.inf_url = inf_url
        self.http_client = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=pool_size, max_connections=pool_size)
        )
        # Only support clip models for now
        self.preprocess = _get_transform(224)
        self.tokenise = open_clip.get_tokenizer('ViT-B-16-SigLIP')

    @staticmethod
    def tensor_to_json(tensor: torch.Tensor) -> dict:
        return {
            'dtype': tensor.dtype,
            'shape': tensor.shape,
            'data': tensor.tolist()
        }

    @staticmethod
    def create_vectorize_request(metadata: VectoriseRequest, tensor):
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)

        files = {
            'metadata': (None, orjson.dumps(metadata.dict()), 'application/json'),
            'tensor_file': ('tensor.pt', buffer, 'application/octet-stream')
        }
        return files

    def vectorise(self, model_name: str, content: Union[str, List[str]],
                  model_properties: dict = None, device: str = None, normalize_embeddings: bool = True,
                  model_auth: ModelAuth = None, enable_cache: bool = False, modality: Modality = Modality.TEXT,
                  media_download_headers: Optional[dict] = None, **kwargs) -> List[List[float]]:

        # Preprocess
        if modality == Modality.TEXT:
            with RequestMetricsStore.for_request().time("inference.preprocess"):
                tensor: torch.Tensor = self.tokenise(content)
        elif modality == Modality.IMAGE:
            if isinstance(content, list):
                # we only support 1 image now
                content = content[0]
            with RequestMetricsStore.for_request().time("inference.download"):
                image = format_and_load_CLIP_image(content, media_download_headers)
            with RequestMetricsStore.for_request().time("inference.preprocess"):
                tensor: torch.Tensor = self.preprocess(image)
        else:
            raise InferenceError(f'Preprocessing for modality {modality} is not supported.')

        endpoint = f'{self.inf_url}/vectorise-binary'

        request = VectoriseRequest(
            model_name=model_name,
            model_properties=model_properties,
            model_auth=model_auth,
            modality=modality,
            normalize_embeddings=normalize_embeddings,
            device=device,
            enable_cache=enable_cache,
            media_download_headers=media_download_headers,
            # we do not populate the content since it's already preprocessed to tensor.
            # We attach the tensor in a binary file instead
            preprocessed=True,
            # TODO support other params
        )

        try:
            with RequestMetricsStore.for_request().time("inference.roundtrip"):
                resp = self.http_client.post(endpoint, files=self.create_vectorize_request(request, tensor))
        except httpx.HTTPError as e:
            raise InferenceError(e) from e

        resp.raise_for_status()

        result = VectoriseResponse.construct(**orjson.loads(resp.text))
        RequestMetricsStore.for_request().add_time("inference.local_vectorise", result.vectorise_time)

        return result.embeddings


bypassing_inference = os.environ.get("MARQO_BYPASS_INFERENCE", "FALSE") == 'TRUE'
running_remote_inference = utils.read_env_vars_and_defaults(EnvVars.MARQO_REMOTE_INFERENCE) == 'TRUE'
slim_inference = os.environ.get("MARQO_SLIM_INFERENCE", "FALSE") == 'TRUE'

if bypassing_inference:
    vectoriser = StaticVectoriser(dimension=768)
elif running_remote_inference:
    remote_inference_url = utils.read_env_vars_and_defaults(EnvVars.MARQO_REMOTE_INFERENCE_URL)

    if slim_inference:
        vectoriser = RemoteSlimVectoriser(inf_url=remote_inference_url)
    else:
        vectoriser = RemoteVectoriser(inf_url=remote_inference_url)
else:
    vectoriser = LocalVectoriser()


def vectorise(model_name: str, content: Union[str, List[str]],
              model_properties: dict = None,
              device: str = None, normalize_embeddings: bool = True,
              model_auth: ModelAuth = None, enable_cache: bool = False, modality: Modality = Modality.TEXT,
              media_download_headers: Optional[dict] = None, **kwargs) -> List[List[float]]:

    return vectoriser.vectorise(model_name=model_name, content=content, model_properties=model_properties,
                                  device=device, normalize_embeddings=normalize_embeddings, model_auth=model_auth,
                                  enable_cache=enable_cache, modality=modality,
                                  media_download_headers=media_download_headers, **kwargs)

