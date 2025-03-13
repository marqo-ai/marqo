import json
from typing import List

import pydantic
import uvicorn
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
from marqo.api.models.add_docs_objects import AddDocsBodyParams
from marqo.tensor_search.models.api_models import SearchQuery
from marqo.tensor_search.models.index_settings import IndexSettings, IndexSettingsWithName
from marqo.tensor_search.on_start_script import on_start
from marqo.tensor_search.telemetry import RequestMetricsStore, TelemetryMiddleware
from marqo.tensor_search.throttling.redis_throttle import throttle
from marqo.tensor_search.web import api_validation, api_utils
from marqo.upgrades.upgrade import UpgradeRunner, RollbackRunner
from marqo.vespa import exceptions as vespa_exceptions
from marqo.vespa.vespa_client import VespaClient
from marqo.vespa.zookeeper_client import ZookeeperClient


app = FastAPI(
    title="Marqo Native inference API",
    version=version.get_version()
)


@app.post("inference")
