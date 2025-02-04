"""The API entrypoint for Tensor Search"""
import json
import os
from contextlib import asynccontextmanager
from typing import List

import pydantic
from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, ORJSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from marqo import config, marqo_docs
from marqo import exceptions as base_exceptions
from marqo import version
from marqo.api import exceptions as api_exceptions
from marqo.api.exceptions import InvalidArgError, UnprocessableEntityError
from marqo.api.models.add_docs_objects import AddDocsBodyParams
from marqo.api.models.embed_request import EmbedRequest
from marqo.api.models.health_response import HealthResponse
from marqo.api.models.recommend_query import RecommendQuery
from marqo.api.models.rollback_request import RollbackRequest
from marqo.api.models.update_documents import UpdateDocumentsBodyParams
from marqo.api.route import MarqoCustomRoute
from marqo.core import exceptions as core_exceptions
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.monitoring import memory_profiler
from marqo.logging import get_logger
from marqo.tensor_search import tensor_search, utils
from marqo.tensor_search.enums import RequestType, EnvVars
from marqo.tensor_search.main import get_config
from marqo.tensor_search.models.api_models import SearchQuery
from marqo.tensor_search.models.index_settings import IndexSettings, IndexSettingsWithName
from marqo.tensor_search.on_start_script import on_start, StartMode
from marqo.tensor_search.telemetry import RequestMetricsStore, TelemetryMiddleware
from marqo.tensor_search.throttling.redis_throttle import throttle
from marqo.tensor_search.web import api_validation, api_utils
from marqo.upgrades.upgrade import UpgradeRunner, RollbackRunner
from marqo.vespa import exceptions as vespa_exceptions
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = get_logger(__name__)


logger.info(f'{os.getpid()}: {__name__} on_start')
will_run_remote_inference = utils.read_env_vars_and_defaults(EnvVars.MARQO_REMOTE_INFERENCE) == 'TRUE'
start_mode = StartMode.API if will_run_remote_inference else (StartMode.API | StartMode.INFERENCE)
on_start(get_config(), start_mode)


# Middleware to capture a specific header (e.g., 'X-Request-ID')
class CustomHeaderLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Extract the custom header, for example, 'X-Request-ID'
        request_pid = request.headers.get('X-Request-PID', 'na')
        request.state.request_pid = request_pid
        response = await call_next(request)
        return response


app = FastAPI(
    title="Marqo",
    version=version.get_version(),
)
app.add_middleware(TelemetryMiddleware)
app.add_middleware(CustomHeaderLoggingMiddleware)
app.router.route_class = MarqoCustomRoute


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
        (core_exceptions.InvalidFieldNameError, api_exceptions.InvalidFieldNameError, None, None),
        (core_exceptions.IndexExistsError, api_exceptions.IndexAlreadyExistsError, None, None),
        (core_exceptions.IndexNotFoundError, api_exceptions.IndexNotFoundError, None, None),
        (core_exceptions.VespaDocumentParsingError, api_exceptions.BackendDataParsingError, None, None),
        (core_exceptions.OperationConflictError, api_exceptions.OperationConflictError, None, None),
        (core_exceptions.BackendCommunicationError, api_exceptions.BackendCommunicationError, None, None),
        (core_exceptions.ZeroMagnitudeVectorError, api_exceptions.BadRequestError, None, None),
        (core_exceptions.BackendCommunicationError, api_exceptions.BackendCommunicationError, None, None),
        (core_exceptions.ModelError, api_exceptions.BadRequestError, None, marqo_docs.list_of_models()),
        (core_exceptions.UnsupportedFeatureError, api_exceptions.BadRequestError, None, None),
        (core_exceptions.InternalError, api_exceptions.InternalError, None, None),
        (core_exceptions.ApplicationRollbackError, api_exceptions.ApplicationRollbackError, None, None),
        (core_exceptions.TooManyFieldsError, api_exceptions.BadRequestError, None, None),

        # Vespa client exceptions
        (
            vespa_exceptions.VespaTimeoutError,
            api_exceptions.VectorStoreTimeoutError,
            "Vector store request timed out. Try your request again later.",
            None
        ),

        # Base exceptions
        (base_exceptions.InternalError, api_exceptions.InternalError, None, None),
        (base_exceptions.InvalidArgumentError, api_exceptions.InvalidArgError, None, None),
    ]

    converted_error = None
    for base_exception, api_exception, message, link in api_exception_mappings:
        if isinstance(exc, base_exception):
            error_message = message or exc.message
            converted_error = api_exception(message=error_message, link=link)
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


@app.exception_handler(RequestValidationError)
async def api_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Catch FastAPI validation errors and return a 422 error with the error messages.

    Note: The Pydantic Validation error that happens at the API will be caught here and returned as a 422 error.
    However, the Pydantic Validation error that happens in the core will be caught by the MarqoError handler above and
    converted to an API error in validation_exception_handler
    """
    body = {
        "detail": jsonable_encoder(exc.errors()),
        "code": UnprocessableEntityError.code,
        "type": UnprocessableEntityError.error_type,
        "link": UnprocessableEntityError.link
    }
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content=body
    )


@app.exception_handler(pydantic.ValidationError)
async def validation_exception_handler(request, exc: pydantic.ValidationError) -> JSONResponse:
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


@app.on_event("shutdown")
def shutdown_event():
    """Close the Zookeeper client on shutdown."""
    marqo_config = get_config()
    marqo_config.stop_and_close_zookeeper_client()


@app.get("/")
def root():
    return {"message": "Welcome to Marqo",
            "version": version.get_version()}


@app.get('/memory')
@utils.enable_debug_apis()
def memory():
    return memory_profiler.get_memory_profile()


@app.post('/validate/index/{index_name}')
@utils.enable_ops_api()
def schema_validation(index_name: str, settings_object: dict):
    IndexManagement.validate_index_settings(index_name, settings_object)

    return JSONResponse(
        content={
            "validated": True,
            "index": index_name
        }
    )


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
        result = tensor_search.search(
            config=marqo_config, text=search_query.q,
            index_name=index_name, highlights=search_query.showHighlights,
            searchable_attributes=search_query.searchableAttributes,
            search_method=search_query.searchMethod,
            result_count=search_query.limit, offset=search_query.offset,
            ef_search=search_query.efSearch, approximate=search_query.approximate,
            reranker=search_query.reRanker,
            filter=search_query.filter, device=device,
            attributes_to_retrieve=search_query.attributesToRetrieve, boost=search_query.boost,
            media_download_headers = search_query.mediaDownloadHeaders,
            context=search_query.context,
            score_modifiers=search_query.scoreModifiers,
            model_auth=search_query.modelAuth,
            text_query_prefix=search_query.textQueryPrefix,
            hybrid_parameters=search_query.hybridParameters
        )
        return ORJSONResponse(result)


@app.post("/indexes/{index_name}/recommend")
@throttle(RequestType.SEARCH)
def recommend(query: RecommendQuery, index_name: str,
              marqo_config: config.Config = Depends(get_config)):
    with RequestMetricsStore.for_request().time(f"POST /indexes/{index_name}/search"):
        return marqo_config.recommender.recommend(
            index_name=index_name,
            documents=query.documents,
            tensor_fields=query.tensorFields,
            interpolation_method=query.interpolationMethod,
            exclude_input_documents=query.excludeInputDocuments,
            result_count=query.limit,
            offset=query.offset,
            highlights=query.showHighlights,
            ef_search=query.efSearch,
            approximate=query.approximate,
            searchable_attributes=query.searchableAttributes,
            reranker=query.reRanker,
            filter=query.filter,
            attributes_to_retrieve=query.attributesToRetrieve,
            score_modifiers=query.scoreModifiers
        )


@app.post("/indexes/{index_name}/documents")
@throttle(RequestType.INDEX)
def add_or_replace_documents(
        body: AddDocsBodyParams,
        index_name: str,
        marqo_config: config.Config = Depends(get_config),
        device: str = Depends(api_validation.validate_device)):
    """add_documents endpoint (replace existing docs with the same id)"""
    add_docs_params = api_utils.add_docs_params_orchestrator(index_name=index_name, body=body,
                                                             device=device)

    with RequestMetricsStore.for_request().time(f"POST /indexes/{index_name}/documents"):
        res = marqo_config.document.add_documents(add_docs_params=add_docs_params)
        return JSONResponse(content=res.dict(exclude_none=True, by_alias=True), headers=res.get_header_dict())


@app.post("/indexes/{index_name}/embed")
@throttle(RequestType.SEARCH)
def embed(embedding_request: EmbedRequest, index_name: str, device: str = Depends(api_validation.validate_device),
          marqo_config: config.Config = Depends(get_config)):
    with RequestMetricsStore.for_request().time(f"POST /indexes/{index_name}/embed"):
        return marqo_config.embed.embed_content(
            content=embedding_request.content,
            index_name=index_name, device=device,
            media_download_headers=embedding_request.mediaDownloadHeaders,
            model_auth=embedding_request.modelAuth,
            content_type=embedding_request.content_type
        )


@app.patch("/indexes/{index_name}/documents")
@throttle(RequestType.PARTIAL_UPDATE)
def update_documents(
        body: UpdateDocumentsBodyParams,
        index_name: str,
        marqo_config: config.Config = Depends(get_config)):
    """update_documents endpoint"""

    res = marqo_config.document.partial_update_documents_by_index_name(
        index_name=index_name, partial_documents=body.documents)

    return JSONResponse(content=res.dict(exclude_none=True, by_alias=True), headers=res.get_header_dict())


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
    res = tensor_search.get_documents_by_ids(
        config=marqo_config, index_name=index_name, document_ids=document_ids,
        show_vectors=expose_facets
    )
    return JSONResponse(content=res.dict(exclude_none=True, by_alias=True), headers=res.get_header_dict())


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
def get_cuda_info(marqo_config: config.Config = Depends(get_config)):
    return marqo_config.monitoring.get_cuda_info()


@app.post("/batch/indexes/delete")
@utils.enable_batch_apis()
def batch_delete_indexes(index_names: List[str], marqo_config: config.Config = Depends(get_config)):
    """An internal API used for testing processes. Not to be used by users."""
    marqo_config.index_management.batch_delete_indexes_by_name(index_names=index_names)
    return JSONResponse(content={"acknowledged": True,
                                 "index_names": index_names}, status_code=200)


@app.post("/batch/indexes/create")
@utils.enable_batch_apis()
def batch_create_indexes(index_settings_with_name_list: List[IndexSettingsWithName],
                         marqo_config: config.Config = Depends(get_config)):
    """An internal API used for testing processes. Not to be used by users."""

    marqo_index_requests = [settings.to_marqo_index_request(settings.indexName) for
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
    document_count: int = marqo_config.document.delete_all_docs_by_index_name(index_name=index_name)

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


@app.post("/rollback-vespa")
def rollback_vespa_app_to_current_version(marqo_config: config.Config = Depends(get_config)):
    marqo_config.index_management.rollback_vespa()
    return JSONResponse(
        content={"version": version.get_version()},
        status_code=200
    )


