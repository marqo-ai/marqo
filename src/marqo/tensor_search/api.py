"""The API entrypoint for Tensor Search"""
import json
import os
from typing import List

import pydantic
import uvicorn
from fastapi import FastAPI
from fastapi import Request, Depends
from fastapi.responses import JSONResponse

from marqo import config, errors
from marqo import version
from marqo.core.exceptions import IndexExistsError, IndexNotFoundError
from marqo.core.index_management.index_management import IndexManagement
from marqo.errors import InvalidArgError, MarqoWebError, MarqoError
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import RequestType, EnvVars
from marqo.tensor_search.models.add_docs_objects import (AddDocsBodyParams)
from marqo.tensor_search.models.api_models import SearchQuery
from marqo.tensor_search.models.index_settings import IndexSettings
from marqo.tensor_search.on_start_script import on_start
from marqo.tensor_search.telemetry import RequestMetricsStore, TelemetryMiddleware
from marqo.tensor_search.throttling.redis_throttle import throttle
from marqo.tensor_search.web import api_validation, api_utils
from marqo.vespa.vespa_client import VespaClient


def generate_config() -> config.Config:
    vespa_client = VespaClient(
        config_url=os.environ[EnvVars.VESPA_CONFIG_URL],
        query_url=os.environ[EnvVars.VESPA_QUERY_URL],
        document_url=os.environ[EnvVars.VESPA_DOCUMENT_URL],
        pool_size=os.environ.get(EnvVars.VESPA_POOL_SIZE, 10),
    )
    index_management = IndexManagement(vespa_client)
    return config.Config(vespa_client, index_management)


_config = generate_config()

if __name__ in ["__main__", "api"]:
    on_start(_config)

app = FastAPI(
    title="Marqo",
    version=version.get_version()
)
app.add_middleware(TelemetryMiddleware)


def get_config():
    return _config


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
def create_index(index_name: str, settings: IndexSettings, marqo_config: config.Config = Depends(get_config)):
    try:
        marqo_config.index_management.create_index(settings.to_marqo_index_request(index_name))
    except IndexExistsError as e:
        raise errors.IndexAlreadyExistsError(f"Index {index_name} already exists") from e

    return JSONResponse(
        content={
            "acknowledged": True,
            "index": index_name
        },
        status_code=200
    )


@app.post("/indexes/{index_name}/search")
@throttle(RequestType.SEARCH)
def search(search_query: SearchQuery, index_name: str, device: str = Depends(api_validation.validate_device),
           marqo_config: config.Config = Depends(get_config)):
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
            image_download_headers=search_query.image_download_headers,
            context=search_query.context,
            score_modifiers=search_query.scoreModifiers,
            model_auth=search_query.modelAuth
        )


@app.post("/indexes/{index_name}/documents")
@throttle(RequestType.INDEX)
def add_or_replace_documents(
        body: AddDocsBodyParams,
        index_name: str,
        refresh: bool = True,
        marqo_config: config.Config = Depends(get_config),
        device: str = Depends(api_validation.validate_device)):
    """add_documents endpoint (replace existing docs with the same id)"""
    add_docs_params = api_utils.add_docs_params_orchestrator(index_name=index_name, body=body,
                                                             device=device, auto_refresh=refresh)

    with RequestMetricsStore.for_request().time(f"POST /indexes/{index_name}/documents"):
        return tensor_search.add_documents(
            config=marqo_config, add_docs_params=add_docs_params
        )


@app.get("/indexes/{index_name}/documents/{document_id}")
def get_document_by_id(index_name: str, document_id: str,
                       marqo_config: config.Config = Depends(get_config),
                       expose_facets: bool = False):
    return tensor_search.get_document_by_id(
        config=marqo_config, index_name=index_name, document_id=document_id,
        show_vectors=expose_facets
    )


@app.get("/indexes/{index_name}/documents")
def get_documents_by_ids(
        index_name: str, document_ids: List[str],
        marqo_config: config.Config = Depends(get_config),
        expose_facets: bool = False):
    return tensor_search.get_documents_by_ids(
        config=marqo_config, index_name=index_name, document_ids=document_ids,
        show_vectors=expose_facets
    )


@app.get("/indexes/{index_name}/stats")
def get_index_stats(index_name: str, marqo_config: config.Config = Depends(get_config)):
    stats = marqo_config.monitoring.get_index_stats(index_name)
    return JSONResponse(
        content={
            'numberOfDocuments': stats.number_of_documents
        },
        status_code=200
    )


@app.delete("/indexes/{index_name}")
def delete_index(index_name: str, marqo_config: config.Config = Depends(get_config)):
    tensor_search.delete_index(index_name=index_name, config=marqo_config)

    return JSONResponse(content={"acknowledged": True}, status_code=200)


@app.post("/indexes/{index_name}/documents/delete-batch")
def delete_docs(index_name: str, documentIds: List[str], refresh: bool = True,
                marqo_config: config.Config = Depends(get_config)):
    return tensor_search.delete_documents(
        index_name=index_name, config=marqo_config, doc_ids=documentIds,
        auto_refresh=refresh
    )


@app.get("/health")
def check_health(marqo_config: config.Config = Depends(get_config)):
    return marqo_config.monitoring.get_health()


@app.get("/indexes/{index_name}/health")
def check_index_health(index_name: str, marqo_config: config.Config = Depends(get_config)):
    return marqo_config.monitoring.get_health(index_name=index_name)


@app.get("/indexes")
def get_indexes(marqo_config: config.Config = Depends(get_config)):
    indexes = marqo_config.index_management.get_all_indexes()

    return {
        'results': [
            {'index_name': index.name for index in indexes}
        ]
    }


@app.get("/indexes/{index_name}/settings")
def get_settings(index_name: str, marqo_config: config.Config = Depends(get_config)):
    try:
        marqo_index = marqo_config.index_management.get_index(index_name)
        return IndexSettings.from_marqo_index(marqo_index).dict(exclude_none=True)
    except IndexNotFoundError as e:
        raise errors.IndexNotFoundError(f"Index {index_name} not found") from e


@app.get("/models")
def get_loaded_models():
    return tensor_search.get_loaded_models()


@app.delete("/models")
def eject_model(model_name: str, model_device: str):
    return tensor_search.eject_model(model_name=model_name, device=model_device)


@app.get("/device/cpu")
def get_cpu_info():
    return tensor_search.get_cpu_info()


@app.get("/device/cuda")
def get_cuda_info():
    return tensor_search.get_cuda_info()


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8882)

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
