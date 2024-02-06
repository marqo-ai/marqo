"""The API entrypoint for Tensor Search"""
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi import Request, Depends
from fastapi.responses import JSONResponse

from marqo import config
from marqo import exceptions as base_exceptions
from marqo import version
from marqo.api import exceptions as api_exceptions
from marqo.api.models.health_response import HealthResponse
from marqo.api.models.rollback_request import RollbackRequest
from marqo.api.route import MarqoCustomRoute
from marqo.core import exceptions as core_exceptions
from marqo.core.index_management.index_management import IndexManagement
from marqo.logging import get_logger
from marqo.tensor_search import tensor_search, utils
from marqo.tensor_search.enums import RequestType, EnvVars
from marqo.tensor_search.models.add_docs_objects import (AddDocsBodyParams)
from marqo.tensor_search.models.api_models import SearchQuery
from marqo.tensor_search.models.index_settings import IndexSettings, IndexSettingsWithName
from marqo.tensor_search.on_start_script import on_start
from marqo.tensor_search.telemetry import RequestMetricsStore, TelemetryMiddleware
from marqo.tensor_search.throttling.redis_throttle import throttle
from marqo.tensor_search.web import api_validation, api_utils
from marqo.upgrades.upgrade import UpgradeRunner, RollbackRunner
from marqo.vespa.vespa_client import VespaClient

logger = get_logger(__name__)


def generate_config() -> config.Config:
    vespa_client = VespaClient(
        config_url=utils.read_env_vars_and_defaults(EnvVars.VESPA_CONFIG_URL),
        query_url=utils.read_env_vars_and_defaults(EnvVars.VESPA_QUERY_URL),
        document_url=utils.read_env_vars_and_defaults(EnvVars.VESPA_DOCUMENT_URL),
        pool_size=utils.read_env_vars_and_defaults_ints(EnvVars.VESPA_POOL_SIZE),
        content_cluster_name=utils.read_env_vars_and_defaults(EnvVars.VESPA_CONTENT_CLUSTER_NAME),
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
app.router.route_class = MarqoCustomRoute


def get_config():
    return _config


@app.exception_handler(base_exceptions.MarqoError)
def marqo_base_exception_handler(request: Request, exc: base_exceptions.MarqoError) -> JSONResponse:
    """
    Catch a base/core Marqo Error and convert to its corresponding API Marqo Error.
    The API Error will be passed to the `marqo_api_exception_handler` below.
    This ensures that raw base errors are never returned by the API.
    
    Mappings are in an ordered list to allow for hierarchical resolution of errors.
    Stored as 2-tuples: (Base/Core/Vespa/Inference Error, API Error)
    """
    api_exception_mappings = [
        # More specific errors should take precedence

        # Core exceptions
        (core_exceptions.InvalidFieldNameError, api_exceptions.InvalidFieldNameError),
        (core_exceptions.IndexExistsError, api_exceptions.IndexAlreadyExistsError),
        (core_exceptions.IndexNotFoundError, api_exceptions.IndexNotFoundError),
        (core_exceptions.VespaDocumentParsingError, api_exceptions.BackendDataParsingError),

        # Base exceptions
        (base_exceptions.InternalError, api_exceptions.InternalError),
        (base_exceptions.InvalidArgumentError, api_exceptions.InvalidArgError),
    ]

    converted_error = None
    for base_exception, api_exception in api_exception_mappings:
        if isinstance(exc, base_exception):
            converted_error = api_exception(exc.message)
            break

    # Completely unhandled exception (500)
    # This should abstract away internal error.
    if not converted_error:
        converted_error = api_exceptions.MarqoWebError("Marqo encountered an unexpected internal error.")

    return marqo_api_exception_handler(request, converted_error)


@app.exception_handler(api_exceptions.MarqoWebError)
def marqo_api_exception_handler(request: Request, exc: api_exceptions.MarqoWebError) -> JSONResponse:
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


@app.exception_handler(api_exceptions.MarqoError)
def marqo_internal_exception_handler(request, exc: api_exceptions.MarqoError):
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
    marqo_config.index_management.create_index(settings.to_marqo_index_request(index_name))

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
            ef_search=search_query.efSearch, approximate=search_query.approximate,
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
    stats = marqo_config.monitoring.get_index_stats_by_name(index_name)
    return {
        'numberOfDocuments': stats.number_of_documents,
        'numberOfVectors': stats.number_of_vectors,
        'backend': {
            'memoryUsedPercentage': stats.backend.memory_used_percentage,
            'storageUsedPercentage': stats.backend.storage_used_percentage
        }
    }


@app.delete("/indexes/{index_name}")
def delete_index(index_name: str, marqo_config: config.Config = Depends(get_config)):
    tensor_search.delete_index(index_name=index_name, config=marqo_config)

    return JSONResponse(content={"acknowledged": True}, status_code=200)


@app.post("/indexes/{index_name}/documents/delete-batch")
def delete_docs(index_name: str, documentIds: List[str],
                marqo_config: config.Config = Depends(get_config)):
    return tensor_search.delete_documents(
        index_name=index_name, config=marqo_config, doc_ids=documentIds
    )


@app.get("/health")
def check_health(marqo_config: config.Config = Depends(get_config)):
    health_status = marqo_config.monitoring.get_health()
    return HealthResponse.from_marqo_health_status(health_status)


@app.get("/indexes/{index_name}/health")
def check_index_health(index_name: str, marqo_config: config.Config = Depends(get_config)):
    health_status = marqo_config.monitoring.get_health(index_name=index_name)
    return HealthResponse.from_marqo_health_status(health_status)


@app.get("/indexes")
def get_indexes(marqo_config: config.Config = Depends(get_config)):
    indexes = marqo_config.index_management.get_all_indexes()
    return {
        'results': [
            {'indexName': index.name} for index in indexes
        ]
    }


@app.get("/indexes/{index_name}/settings")
def get_settings(index_name: str, marqo_config: config.Config = Depends(get_config)):
    marqo_index = marqo_config.index_management.get_index(index_name)
    return IndexSettings.from_marqo_index(marqo_index).dict(exclude_none=True, by_alias=True)


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


@app.post("/batch/indexes/delete")
@utils.enable_batch_apis()
def batch_delete_indexes(index_names: List[str], marqo_config: config.Config = Depends(get_config)):
    """An internal API used for testing processes. Not to be used by users."""
    marqo_config.index_management.batch_delete_indexes_by_name(index_names=index_names)
    return JSONResponse(content={"acknowledged": True,
                                 "index_names": index_names}, status_code=200)


@app.post("/batch/indexes/create")
@utils.enable_batch_apis()
def batch_create_indexes(index_settings_with_name_list: List[IndexSettingsWithName], \
                         marqo_config: config.Config = Depends(get_config)):
    """An internal API used for testing processes. Not to be used by users."""

    marqo_index_requests = [settings.to_marqo_index_request(settings.indexName) for \
                            settings in index_settings_with_name_list]

    marqo_config.index_management.batch_create_indexes(marqo_index_requests)

    return JSONResponse(
        content={
            "acknowledged": True,
            "index_names": [settings.indexName for settings in index_settings_with_name_list]
        },
        status_code=200
    )


@app.delete("/indexes/{index_name}/documents/delete-all")
@utils.enable_batch_apis()
def delete_all_documents(index_name: str, marqo_config: config.Config = Depends(get_config)):
    """An internal API used for testing processes. Not to be used by users.
    This API delete all the documents in the indexes specified in the index_names list."""
    document_count: int = marqo_config.document.delete_all_docs(index_name=index_name)

    return {"documentCount": document_count}


@app.post("/upgrade")
@utils.enable_upgrade_api()
def upgrade_marqo(marqo_config: config.Config = Depends(get_config)):
    """An internal API used for testing processes. Not to be used by users."""
    upgrade_runner = UpgradeRunner(marqo_config.vespa_client, marqo_config.index_management)
    upgrade_runner.upgrade()


@app.post("/rollback")
@utils.enable_upgrade_api()
def rollback_marqo(req: RollbackRequest, marqo_config: config.Config = Depends(get_config)):
    """An internal API used for testing processes. Not to be used by users."""
    rollback_runner = RollbackRunner(marqo_config.vespa_client, marqo_config.index_management)
    rollback_runner.rollback(from_version=req.from_version, to_version=req.to_version)


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
