from typing import List, Union, Protocol, Optional

import httpx
from orjson import orjson
import numpy as np
import os

from marqo.s2_inference import s2_inference
from marqo.s2_inference.multimodal_model_load import Modality
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.models.inf_request import VectoriseRequest
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

        return orjson.loads(resp.text)


bypassing_inference = os.environ.get("MARQO_BYPASS_INFERENCE", "FALSE") == 'TRUE'
running_remote_inference = utils.read_env_vars_and_defaults(EnvVars.MARQO_REMOTE_INFERENCE) == 'TRUE'

if bypassing_inference:
    vectoriser = StaticVectoriser(dimension=768)
elif running_remote_inference:
    remote_inference_url = utils.read_env_vars_and_defaults(EnvVars.MARQO_REMOTE_INFERENCE_URL)
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

