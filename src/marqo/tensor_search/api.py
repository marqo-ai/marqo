"""The API entrypoint for Tensor Search"""
import typing
from fastapi.responses import JSONResponse
from models.api_models import SearchQuery
from fastapi import FastAPI, Request, Depends, HTTPException
from marqo.errors import MarqoWebError, MarqoError
from fastapi import FastAPI, Query
from marqo.tensor_search import tensor_search
from marqo import config
from typing import List, Dict
import os
from marqo.tensor_search.web import api_validation, api_utils
from marqo.tensor_search import utils
from marqo.tensor_search.on_start_script import on_start
from marqo import version

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


OPENSEARCH_URL = replace_host_localhosts(
    os.environ.get("OPENSEARCH_IS_INTERNAL", None),
    os.environ["OPENSEARCH_URL"])


on_start(OPENSEARCH_URL)
app = FastAPI(
    title="Marqo",
    version=version.get_version()
)


def generate_config() -> config.Config:
    return config.Config(api_utils.upconstruct_authorized_url(
        opensearch_url=OPENSEARCH_URL
    ))


@app.exception_handler(MarqoWebError)
def marqo_user_exception_handler(request, exc: MarqoWebError):
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


@app.post("/indexes/{index_name}/search")
def search(search_query: SearchQuery, index_name: str, device: str = Depends(api_validation.validate_device),
                 marqo_config: config.Config = Depends(generate_config)):
    return tensor_search.search(
        config=marqo_config, text=search_query.q,
        index_name=index_name, highlights=search_query.showHighlights,
        searchable_attributes=search_query.searchableAttributes,
        search_method=search_query.searchMethod,
        result_count=search_query.limit, reranker=search_query.reRanker,
        filter=search_query.filter, device=device,
        attributes_to_retrieve=search_query.attributesToRetrieve
    )


@app.post("/indexes/{index_name}/documents")
def add_or_replace_documents(docs: List[Dict], index_name: str, refresh: bool = True,
                        marqo_config: config.Config = Depends(generate_config),
                        batch_size: int = 0, processes: int = 1,
                        non_tensor_fields: List[str] = Query(default=[]),
                        device: str = Depends(api_validation.validate_device)):
    """add_documents endpoint (replace existing docs with the same id)"""
    return tensor_search.add_documents_orchestrator(
        config=marqo_config,
        docs=docs,
        index_name=index_name, auto_refresh=refresh,
        batch_size=batch_size, processes=processes, device=device,
        non_tensor_fields=non_tensor_fields, update_mode='replace'
    )


@app.put("/indexes/{index_name}/documents")
def add_or_update_documents(docs: List[Dict], index_name: str, refresh: bool = True,
                        marqo_config: config.Config = Depends(generate_config),
                        batch_size: int = 0, processes: int = 1,
                        non_tensor_fields: List[str] = Query(default=[]),
                        device: str = Depends(api_validation.validate_device)):
    """update add_documents endpoint"""
    return tensor_search.add_documents_orchestrator(
        config=marqo_config,
        docs=docs,
        index_name=index_name, auto_refresh=refresh,
        batch_size=batch_size, processes=processes, device=device,
        non_tensor_fields=non_tensor_fields, update_mode='update'
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
def delete_docs(index_name: str, documentIds: List[str], refresh: bool = True,
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


@app.get("/indexes")
def get_indexes(marqo_config: config.Config = Depends(generate_config)):
    return tensor_search.get_indexes(config=marqo_config)


@app.get("/models")
def get_loaded_models():
    return tensor_search.get_loaded_models()


@app.delete("/models/eject-model")
def eject_model(model_name: str, device: str, model_info: List[str] = Query(default = [])):
    return tensor_search.eject_model(model_name = model_info["model-name"], model_device = model_name["model-device"])

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

# check the loaded models
"""
curl -XGET http://localhost:8882/models
"""

# eject a model
"""
curl -XDELETE http://localhost:8882/models/eject-model?model-name=ViT-L/14?model-device=cpu
"""