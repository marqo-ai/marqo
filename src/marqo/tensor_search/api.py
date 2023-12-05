"""The API entrypoint for Tensor Search"""

import warnings
from marqo.tensor_search.tensor_search_logging import get_logger
import logging


if get_logger(__name__).getEffectiveLevel() > logging.DEBUG:
    # We need to suppress this warning before the dependency is imported
    warnings.filterwarnings(
        "ignore",
        "Importing `GenerationMixin` from `src/transformers/generation_utils.py` "
        "is deprecated and will be removed in Transformers v5. "
        "Import as `from transformers import GenerationMixin` instead."
    )
    warnings.filterwarnings(
        "ignore",
        ".*Unverified HTTPS request is being made to host 'localhost'.*"
    )
    warnings.filterwarnings(
        "ignore",
        ".*Unverified HTTPS request is being made to host 'host.docker.internal'.*"
    )

import json
import os
import typing
from typing import List, Dict, Optional

import pydantic
from fastapi import FastAPI, Query
from fastapi import Request, Depends
from fastapi.responses import JSONResponse

from marqo import config
from marqo import version
from marqo.errors import InvalidArgError, MarqoWebError, MarqoError, BadRequestError
from marqo.tensor_search import tensor_search
from marqo.tensor_search.backend import get_index_info
from marqo.tensor_search.enums import RequestType
from marqo.tensor_search.models.add_docs_objects import (AddDocsParams, ModelAuth,
                                                         AddDocsBodyParams)
from marqo.tensor_search.models.api_models import BulkSearchQuery, SearchQuery
from marqo.tensor_search.on_start_script import on_start
from marqo.tensor_search.telemetry import RequestMetricsStore, TelemetryMiddleware
from marqo.tensor_search.throttling.redis_throttle import throttle
from marqo.tensor_search.utils import add_timing
from marqo.tensor_search.web import api_validation, api_utils


def replace_host_localhosts(OPENSEARCH_IS_INTERNAL: str, OS_URL: str):
    """Replaces a host's localhost URL with one that can be referenced from
    within a container.

    If we are within a docker container, we need to determine if a localhost
    OpenSearch URL is referring to a URL within our container, or a URL on the
    host.

    Note this only works if the Docker run command is ran with this option:
    --add-host host.docker.internal:host-gateway

    Args:
        OPENSEARCH_IS_INTERNAL: 'False' | 'True'; these are strings because they
            come from environment vars
        OS_URL: the URL of OpenSearch
    """
    if OPENSEARCH_IS_INTERNAL == "False":
        for local_domain in ["localhost", "0.0.0.0", "127.0.0.1"]:
            replaced_str = OS_URL.replace(local_domain, "host.docker.internal")
            if replaced_str != OS_URL:
                return replaced_str
    return OS_URL

if __name__ in ["__main__", "api"]:
    OPENSEARCH_URL = replace_host_localhosts(
        os.environ.get("OPENSEARCH_IS_INTERNAL", None),
        os.environ["OPENSEARCH_URL"]
    )
    on_start(OPENSEARCH_URL)

app = FastAPI(
    title="Marqo",
    version=version.get_version()
)
app.add_middleware(TelemetryMiddleware)


def generate_config() -> config.Config:
    return config.Config(api_utils.upconstruct_authorized_url(
        opensearch_url=OPENSEARCH_URL
    ))


@app.exception_handler(MarqoWebError)
def marqo_user_exception_handler(request: Request, exc: MarqoWebError) -> JSONResponse:
    """ Catch a MarqoWebError and return an appropriate HTTP response.

    We can potentially catch any type of Marqo exception. We can do isinstance() calls
    to handle WebErrors vs Regular errors"""
    headers = getattr(exc, "headers", None)
    body = {
        "message": exc.message,
        "code": exc.code,
        "type": exc.error_type,
        "link": exc.link
    }
    if headers:
        return JSONResponse(
            content=body, status_code=exc.status_code, headers=headers
        )
    else:
        return JSONResponse(content=body, status_code=exc.status_code)


@app.exception_handler(pydantic.ValidationError)
async def validation_exception_handler(request: Request, exc: pydantic.ValidationError) -> JSONResponse:
    """Catch pydantic validation errors and rewrite as an InvalidArgError whilst keeping error messages from the ValidationError."""
    error_messages = [{
        'loc': error.get('loc', ''),
        'msg': error.get('msg', ''),
        'type': error.get('type', '')
    } for error in exc.errors()]

    body = {
        "message": json.dumps(error_messages),
        "code": InvalidArgError.code,
        "type": InvalidArgError.error_type,
        "link": InvalidArgError.link
    }
    return JSONResponse(content=body, status_code=InvalidArgError.status_code)

@app.exception_handler(MarqoError)
def marqo_internal_exception_handler(request, exc: MarqoError):
    """MarqoErrors are treated as internal errors"""
    headers = getattr(exc, "headers", None)
    body = {
        "message": exc.message,
        "code": 500,
        "type": "internal_error",
        "link": ""
    }
    if headers:
        return JSONResponse(content=body, status_code=500, headers=headers)
    else:
        return JSONResponse(content=body, status_code=500)


@app.get("/")
def root():
    return {"message": "Welcome to Marqo",
            "version": version.get_version()}


@app.post("/indexes/{index_name}")
def create_index(index_name: str, settings: Dict = None, marqo_config: config.Config = Depends(generate_config)):
    index_settings = dict() if settings is None else settings
    return tensor_search.create_vector_index(
        config=marqo_config, index_name=index_name, index_settings=index_settings
    )


@app.post("/indexes/bulk/search")
@throttle(RequestType.SEARCH)
@add_timing
def bulk_search(query: BulkSearchQuery, device: str = Depends(api_validation.validate_device), marqo_config: config.Config = Depends(generate_config)):
    with RequestMetricsStore.for_request().time(f"POST /indexes/bulk/search"):
        return tensor_search.bulk_search(query, marqo_config, device=device)

@app.post("/indexes/{index_name}/search")
@throttle(RequestType.SEARCH)
def search(search_query: SearchQuery, index_name: str, device: str = Depends(api_validation.validate_device),
           marqo_config: config.Config = Depends(generate_config)):
    
    with RequestMetricsStore.for_request().time(f"POST /indexes/{index_name}/search"):
        return tensor_search.search(
            config=marqo_config, text=search_query.q,
            index_name=index_name, highlights=search_query.showHighlights,
            searchable_attributes=search_query.searchableAttributes,
            search_method=search_query.searchMethod,
            result_count=search_query.limit, offset=search_query.offset,
            reranker=search_query.reRanker, 
            filter=search_query.filter, device=device,
            attributes_to_retrieve=search_query.attributesToRetrieve, boost=search_query.boost,
            download_headers=search_query.download_headers,
            context=search_query.context,
            score_modifiers=search_query.scoreModifiers,
            model_auth=search_query.modelAuth,
            text_query_prefix=search_query.textQueryPrefix
        )



@app.post("/indexes/{index_name}/documents")
@throttle(RequestType.INDEX)
def add_or_replace_documents(
        request: Request,
        body: typing.Union[AddDocsBodyParams, List[Dict]],
        index_name: str,
        refresh: bool = False,
        marqo_config: config.Config = Depends(generate_config),
        non_tensor_fields: Optional[List[str]] = Query(default=None),
        device: str = Depends(api_validation.validate_device),
        use_existing_tensors: Optional[bool] = False,
        download_headers: Optional[dict] = Depends(
            api_utils.decode_download_headers
        ),
        model_auth: Optional[ModelAuth] = Depends(
            api_utils.decode_query_string_model_auth
        ),
        mappings: Optional[dict] = Depends(api_utils.decode_mappings)):

    """add_documents endpoint (replace existing docs with the same id)"""
    add_docs_params = api_utils.add_docs_params_orchestrator(index_name=index_name, body=body,
                                                             device=device, auto_refresh=refresh,
                                                             non_tensor_fields=non_tensor_fields, mappings=mappings,
                                                             model_auth=model_auth,
                                                             download_headers=download_headers,
                                                             use_existing_tensors=use_existing_tensors,
                                                             query_parameters=request.query_params)

    with RequestMetricsStore.for_request().time(f"POST /indexes/{index_name}/documents"):
        return tensor_search.add_documents(
            config=marqo_config, add_docs_params=add_docs_params
        )



@app.get("/indexes/{index_name}/documents/{document_id}")
def get_document_by_id(index_name: str, document_id: str,
                             marqo_config: config.Config = Depends(generate_config),
                             expose_facets: bool = False):
    return tensor_search.get_document_by_id(
        config=marqo_config, index_name=index_name, document_id=document_id,
        show_vectors=expose_facets
    )


@app.get("/indexes/{index_name}/documents")
def get_documents_by_ids(
        index_name: str, document_ids: List[str],
        marqo_config: config.Config = Depends(generate_config),
        expose_facets: bool = False):
    return tensor_search.get_documents_by_ids(
        config=marqo_config, index_name=index_name, document_ids=document_ids,
        show_vectors=expose_facets
    )


@app.get("/indexes/{index_name}/stats")
def get_index_stats(index_name: str, marqo_config: config.Config = Depends(generate_config)):
    return tensor_search.get_stats(
        config=marqo_config, index_name=index_name
    )


@app.delete("/indexes/{index_name}")
def delete_index(index_name: str, marqo_config: config.Config = Depends(generate_config)):
    return tensor_search.delete_index(
        config=marqo_config, index_name=index_name
    )

@app.post("/indexes/{index_name}/documents/delete-batch")
def delete_docs(index_name: str, documentIds: List[str], refresh: bool = False,
                      marqo_config: config.Config = Depends(generate_config)):

    return tensor_search.delete_documents(
        index_name=index_name, config=marqo_config, doc_ids=documentIds,
        auto_refresh=refresh
    )


@app.post("/indexes/{index_name}/refresh")
def refresh_index(index_name: str, marqo_config: config.Config = Depends(generate_config)):
    return tensor_search.refresh_index(
        index_name=index_name, config=marqo_config,
    )


@app.get("/health")
def check_health(marqo_config: config.Config = Depends(generate_config)):
    return tensor_search.check_health(config=marqo_config)


@app.get("/indexes/{index_name}/health")
def check_index_health(index_name: str, marqo_config: config.Config = Depends(generate_config)):
    return tensor_search.check_index_health(config=marqo_config, index_name=index_name)


@app.get("/indexes")
def get_indexes(marqo_config: config.Config = Depends(generate_config)):
    return tensor_search.get_indexes(config=marqo_config)


@app.get("/indexes/{index_name}/settings")
def get_settings(index_name: str, marqo_config: config.Config = Depends(generate_config)):
    index_info = get_index_info(config=marqo_config, index_name=index_name)
    return index_info.index_settings


@app.get("/models")
def get_loaded_models():
    return tensor_search.get_loaded_models()


@app.delete("/models")
def eject_model(model_name:str, model_device:str):
    return tensor_search.eject_model(model_name = model_name, device = model_device)


@app.get("/device/cpu")
def get_cpu_info():
    return tensor_search.get_cpu_info()


@app.get("/device/cuda")
def get_cuda_info():
    return tensor_search.get_cuda_info()

# try these curl commands:

# ADD DOCS:
"""
curl -XPOST  'http://localhost:8882/indexes/my-irst-ix/documents?refresh=true&device=cpu' -H 'Content-type:application/json' -d '
[ 
    {
        "Title": "Honey is a delectable food stuff", 
        "Desc" : "some boring description",
        "_id": "honey_facts_119"
    }, {
        "Title": "Space exploration",
        "Desc": "mooooon! Space!!!!",
        "_id": "moon_fact_145"
    }
]'
"""


# SEARCH DOCS
"""
curl -XPOST  'http://localhost:8882/indexes/my-irst-ix/search?device=cuda0' -H 'Content-type:application/json' -d '{
    "q": "what do bears eat?",
    "searchableAttributes": ["Title", "Desc", "other"],
    "limit": 3,    
    "searchMethod": "TENSOR",
    "showHighlights": true,
    "filter": "Desc:(some boring description)",
    "attributesToRetrieve": ["Title"]
}'
"""

# CREATE CUSTOM IMAGE INDEX:
"""
curl -XPOST http://localhost:8882/indexes/my-multimodal-index -H 'Content-type:application/json' -d '{
    "index_defaults": {
      "treat_urls_and_pointers_as_images":true,    
      "model":"ViT-B/32"
    },
    "number_of_shards": 3
}'
"""

# GET DOCUMENT BY ID:
"""
curl -XGET http://localhost:8882/indexes/my-irst-ix/documents/honey_facts_119
"""

# GET index stats
"""
curl -XGET http://localhost:8882/indexes/my-irst-ix/stats
"""

# GET index settings
"""
curl -XGET http://localhost:8882/indexes/my-irst-ix/settings
"""
# POST refresh index
"""
curl -XPOST  http://localhost:8882/indexes/my-irst-ix/refresh
"""

# DELETE docs
"""
curl -XPOST  http://localhost:8882/indexes/my-irst-ix/documents/delete-batch -H 'Content-type:application/json' -d '[
    "honey_facts_119", "moon_fact_145"
]'
"""

# DELETE index
"""
curl -XDELETE http://localhost:8882/indexes/my-irst-ix
"""

# check cpu info
"""
curl -XGET http://localhost:8882/device/cpu
"""

# check cuda info
"""
curl -XGET http://localhost:8882/device/cuda
"""

# check the loaded models
"""
curl -XGET http://localhost:8882/models
"""

# eject a model
"""
curl -X DELETE 'http://localhost:8882/models?model_name=ViT-L/14&model_device=cuda'
curl -X DELETE 'http://localhost:8882/models?model_name=ViT-L/14&model_device=cpu'
curl -X DELETE 'http://localhost:8882/models?model_name=hf/all_datasets_v4_MiniLM-L6&model_device=cuda' 
curl -X DELETE 'http://localhost:8882/models?model_name=hf/all_datasets_v4_MiniLM-L6&model_device=cpu' 
"""