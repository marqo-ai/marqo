"""tensor search logic

API Notes:
    - Some fields beginning with a double underscore "__" are protected and used for our internal purposes.
    - Examples include:
        __field_name
        __field_content
        __doc_chunk_relation
        __chunk_ids
        fields beginning with "__vector_"
    - The "_id" field isn't a real field. It's a way to declare an ID. Internally we use it as the ID
        for the doc. The doc is stored without this field in its body

Notes on search behaviour with caching and searchable attributes:
    The behaviour of lexical search and vector search differs when it comes to
    interactions between the cache and searchable attributes.

    This issue should just occur on the first search when another user adds a
    new field, as the index cache updates in the background during the search.

    Lexical search:
        - Searching an existing but uncached field will return the best result
            (the uncached field will be searched)
        - Searching all fields will return a poor result
            (the uncached field won’t be searched)
    Vector search:
        - Searching an existing but uncached field will return no results (the
            uncached field won’t be searched)
        - Searching all fields will return a poor result (the uncached field
            won’t be searched)

"""
import copy
import json
from collections import defaultdict
from contextlib import ExitStack
from timeit import default_timer as timer
import functools
import pprint
import typing
from marqo.tensor_search.models.private_models import ModelAuth
import uuid
from typing import List, Optional, Union, Iterable, Sequence, Dict, Any, Tuple
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
import numpy as np
from PIL import Image
import marqo.config as config
from marqo.tensor_search.models.delete_docs_objects import MqDeleteDocsRequest
from marqo.tensor_search.enums import (
    Device, MediaType, MlModel, TensorField, SearchMethod, OpenSearchDataType,
    EnvVars, MappingsObjectType, DocumentFieldType
)
from marqo.tensor_search.enums import IndexSettingsField as NsField
from marqo.tensor_search import utils, backend, validation, configs, add_docs, filtering
from marqo.tensor_search.formatting import _clean_doc
from marqo.tensor_search.index_meta_cache import get_cache, get_index_info
from marqo.tensor_search import index_meta_cache
from marqo.tensor_search.models.api_models import BulkSearchQuery, BulkSearchQueryEntity, ScoreModifier
from marqo.tensor_search.models.search import Qidx, JHash, SearchContext, VectorisedJobs, VectorisedJobPointer
from marqo.tensor_search.models.index_info import IndexInfo, get_model_properties_from_index_defaults
from marqo.tensor_search.models.external_apis.abstract_classes import ExternalAuth
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.tensor_search.health import generate_heath_check_response
from marqo.tensor_search.utils import add_timing
from marqo.tensor_search import delete_docs
from marqo.s2_inference.processing import text as text_processor
from marqo.s2_inference.processing import image as image_processor
from marqo.s2_inference.clip_utils import _is_image
from marqo.s2_inference.reranking import rerank
from marqo.s2_inference import s2_inference
import torch.cuda
import psutil
# We depend on _httprequests.py for now, but this may be replaced in the future, as
# _httprequests.py is designed for the client
from marqo._httprequests import HttpRequests
from marqo.config import Config
from marqo import errors
from marqo.s2_inference import errors as s2_inference_errors
import threading
from dataclasses import replace
from marqo.tensor_search.tensor_search_logging import get_logger

logger = get_logger(__name__)


def _get_dimension_from_model_properties(model_properties: dict) -> int:
    """
    Args:
        model_properties: dict containing model properties
    """
    try:
        return validation.validate_model_dimensions(model_properties["dimensions"])
    except KeyError:
        raise errors.InvalidArgError(
            "The given model properties must contain a 'dimensions' key."
        )
    except errors.InternalError as e:
        # This is caused by bad `dimensions` validation.
        raise errors.InvalidArgError(str(e)) from e


def _add_knn_field(ix_settings: dict):
    """
    This adds the OpenSearch knn field to the index's mappings

    Args:
        ix_settings: the index settings
    """
    model_prop = get_model_properties_from_index_defaults(
        index_defaults=(
            ix_settings["mappings"]["_meta"]
            ["index_settings"][NsField.index_defaults]),
        model_name=(
            ix_settings["mappings"]["_meta"]
            ["index_settings"][NsField.index_defaults][NsField.model])
    )

    ix_settings_with_knn = ix_settings.copy()
    ix_settings_with_knn["mappings"]["properties"][TensorField.chunks]["properties"][TensorField.marqo_knn_field] = {
        "type": "knn_vector",
        "dimension": _get_dimension_from_model_properties(model_prop),
        "method": (
            ix_settings["mappings"]["_meta"]
            ["index_settings"][NsField.index_defaults][NsField.ann_parameters]
        )
    }
    return ix_settings_with_knn


def create_vector_index(
        config: Config, index_name: str, media_type: Union[str, MediaType] = MediaType.default,
        refresh_interval: str = "1s", index_settings=None):
    """
    Args:
        media_type: 'text'|'image'
    """
    validation.validate_index_name(index_name)

    if index_settings is not None:
        if NsField.index_defaults in index_settings:
            _check_model_name(index_settings)
        the_index_settings = _autofill_index_settings(index_settings=index_settings)
    else:
        the_index_settings = configs.get_default_index_settings()

    validation.validate_settings_object(settings_object=the_index_settings)

    vector_index_settings = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
                "refresh_interval": refresh_interval,
                "store.hybrid.mmap.extensions": ["nvd", "dvd", "tim", "tip", "dim", "kdd", "kdi", "cfs", "doc", "vec",
                                                 "vex"]
            },
            "number_of_shards": the_index_settings[NsField.number_of_shards],
            "number_of_replicas": the_index_settings[NsField.number_of_replicas],
        },
        "mappings": {
            "_meta": {
                "media_type": media_type,
            },
            "dynamic_templates": [
                {
                    "strings": {
                        "match_mapping_type": "string",
                        "mapping": {
                            "type": "text"
                        }
                    }
                }
            ],
            "properties": {
                TensorField.chunks: {
                    "type": "nested",
                    "properties": {
                        TensorField.field_name: {
                            "type": "keyword"
                        },
                        TensorField.field_content: {
                            "type": "text"
                        },
                    }
                }
            }
        }
    }
    max_marqo_fields = utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_INDEX_FIELDS)

    if max_marqo_fields is not None:
        max_os_fields = _marqo_field_limit_to_os_limit(int(max_marqo_fields))
        vector_index_settings["settings"]["mapping"] = {"total_fields": {"limit": int(max_os_fields)}}

    model_name = the_index_settings[NsField.index_defaults][NsField.model]
    vector_index_settings["mappings"]["_meta"][NsField.index_settings] = the_index_settings
    vector_index_settings["mappings"]["_meta"]["model"] = model_name

    vector_index_settings_with_knn = _add_knn_field(ix_settings=vector_index_settings)

    logger.debug(f"Creating index {index_name} with settings: {vector_index_settings_with_knn}")
    response = HttpRequests(config).put(path=index_name, body=vector_index_settings_with_knn)

    get_cache()[index_name] = IndexInfo(
        model_name=model_name, properties=vector_index_settings_with_knn["mappings"]["properties"].copy(),
        index_settings=the_index_settings
    )
    return response


def _check_model_name(index_settings):
    """Ensures that if model_properties is given, then model_name is given as well
    """
    model_name = index_settings[NsField.index_defaults].get(NsField.model)
    model_properties = index_settings[NsField.index_defaults].get(NsField.model_properties)
    if model_properties is not None and model_name is None:
        raise s2_inference_errors.UnknownModelError(f"No model name found for model_properties={model_properties}")


def _marqo_field_limit_to_os_limit(marqo_index_field_limit: int) -> int:
    """Translates a Marqo Index Field limit (that a Marqo user will set)
    into the equivalent limit for Marqo-OS

    Each Marqo field generates 2 Marqo-OS fields:
        - One for its content
        - One for filtering

    There are also 4 fields that will be generated on a Marqo index, in most
    cases:
        - one for __vector_marqo_knn_field
        - one for the chunks field
        - one for chunk's __field_content
        - one for chunk's __field_name

    Returns:
        The corresponding Marqo-OS limit
    """
    return (marqo_index_field_limit * 2) + 4


def _autofill_index_settings(index_settings: dict):
    """A half-complete index settings will be auto filled"""

    # TODO: validated conflicting settings
    # treat_urls_and_pointers_as_images

    copied_settings = index_settings.copy()
    default_settings = configs.get_default_index_settings()

    copied_settings = utils.merge_dicts(default_settings, copied_settings)

    if NsField.treat_urls_and_pointers_as_images in copied_settings[NsField.index_defaults] and \
            copied_settings[NsField.index_defaults][NsField.treat_urls_and_pointers_as_images] is True \
            and copied_settings[NsField.index_defaults][NsField.model] is None:
        copied_settings[NsField.index_defaults][NsField.model] = MlModel.clip

    # text preprocessing subfields - fills any missing sub-dict fields if some of the first level are present
    for key in list(default_settings[NsField.index_defaults][NsField.text_preprocessing]):
        if key not in copied_settings[NsField.index_defaults][NsField.text_preprocessing] or \
                copied_settings[NsField.index_defaults][NsField.text_preprocessing][key] is None:
            copied_settings[NsField.index_defaults][NsField.text_preprocessing][key] \
                = default_settings[NsField.index_defaults][NsField.text_preprocessing][key]

    # image preprocessing sub fields - fills any missing sub-dict fields
    for key in list(default_settings[NsField.index_defaults][NsField.image_preprocessing]):
        if key not in copied_settings[NsField.index_defaults][NsField.image_preprocessing] or \
                copied_settings[NsField.index_defaults][NsField.image_preprocessing][key] is None:
            copied_settings[NsField.index_defaults][NsField.image_preprocessing][key] \
                = default_settings[NsField.index_defaults][NsField.image_preprocessing][key]

    return copied_settings


def get_stats(config: Config, index_name: str):
    """Returns the number of documents and vectors in the index."""

    body = {
        "size": 0,
        "aggs": {
            "nested_chunks": {
                "nested": {
                    "path": "__chunks"
                },
                "aggs": {
                    "marqo_vector_count": {
                        "value_count": {
                            # This is a key_word field, so it is fast in value_count
                            "field": "__chunks.__field_name"
                        }
                    }
                }
            }
        }
    }

    try:
        doc_count = HttpRequests(config).post(path=F"{index_name}/_count")["count"]
        vector_count = HttpRequests(config).get(path=f"{index_name}/_search", body=body) \
            ["aggregations"]["nested_chunks"]["marqo_vector_count"]["value"]
    except (KeyError, TypeError) as e:
        raise errors.InternalError(f"Marqo received an unexpected response from Marqo-OS. "
                                   f"The expected fields do not exist in the response. Original error message = {e}")
    except (errors.IndexNotFoundError, errors.InvalidIndexNameError):
        raise
    except errors.MarqoWebError as e:
        raise errors.InternalError(f"Marqo encountered an error while communicating with Marqo-OS. "
                                   f"Original error message: {e.message}")

    return {
        "numberOfDocuments": doc_count,
        "numberOfVectors": vector_count,
    }


def _infer_opensearch_data_type(
        sample_field_content: typing.Any) -> Union[OpenSearchDataType, None]:
    """
    Raises:
        Exception if sample_field_content is a dict
    """
    if isinstance(sample_field_content, Sequence) and len(sample_field_content) > 0:
        # OpenSearch requires that all content of an array be the same type.
        # This function doesn't validate.
        to_check = sample_field_content[0]
    else:
        to_check = sample_field_content

    if isinstance(to_check, dict):
        raise errors.MarqoError(
            "Field content can't be an object. An object should not be passed into _infer_opensearch_data_type"
            "to check.")
    elif isinstance(to_check, str):
        return OpenSearchDataType.text
    else:
        return None


def _get_chunks_for_field(field_name: str, doc_id: str, doc):
    # Find the chunks with a specific __field_name in a doc
    # Note: for a chunkless doc (nothing was tensorised) --> doc["_source"]["__chunks"] == []
    return [chunk for chunk in doc["_source"]["__chunks"] if chunk["__field_name"] == field_name]


def add_documents(config: Config, add_docs_params: AddDocsParams):
    """
    Args:
        config: Config object
        add_docs_params: add_documents()'s parameters
    """
    # ADD DOCS TIMER-LOGGER (3)

    RequestMetricsStore.for_request().start("add_documents.processing_before_opensearch")
    start_time_3 = timer()

    if add_docs_params.mappings is not None:
        validation.validate_mappings_object(mappings_object=add_docs_params.mappings)

    t0 = timer()
    bulk_parent_dicts = []

    try:
        index_info = backend.get_index_info(config=config, index_name=add_docs_params.index_name)
        # Retrieve model dimensions from index info
        index_model_dimensions = index_info.get_model_properties()["dimensions"]
    except errors.IndexNotFoundError:
        raise errors.IndexNotFoundError(f"Cannot add documents to non-existent index {add_docs_params.index_name}")

    existing_fields = set(index_info.properties.keys())
    new_fields = set()

    # A dict to store the multimodal_fields and their (child_fields, opensearch_type)
    # dict = {parent_field_1 : set((child_field_1, type), ),
    #      =  parent_field_2 : set((child_field_1, type), )}
    # Check backend to see the differences between multimodal_fields and new_fields
    new_obj_fields = dict()

    unsuccessful_docs = []
    total_vectorise_time = 0
    
    image_repo = {}
    doc_count = len(add_docs_params.docs)
    
    with ExitStack() as exit_stack:
        if index_info.index_settings[NsField.index_defaults][NsField.treat_urls_and_pointers_as_images]:
            with RequestMetricsStore.for_request().time(
                "image_download.full_time",
                lambda t: logger.debug(
                    f"add_documents image download: took {t:.3f}ms to concurrently download "
                    f"images for {doc_count} docs using {add_docs_params.image_download_thread_count} threads"
                )
            ):
                if add_docs_params.tensor_fields and '_id' in add_docs_params.tensor_fields:
                    raise errors.BadRequestError(message="`_id` field cannot be a tensor field.")

                image_repo = exit_stack.enter_context(
                    add_docs.download_images(docs=add_docs_params.docs, thread_count=20,
                                             tensor_fields=add_docs_params.tensor_fields
                                             if add_docs_params.tensor_fields is not None else None,
                                             non_tensor_fields=add_docs_params.non_tensor_fields + ['_id']
                                             if add_docs_params.non_tensor_fields is not None else None,
                                             image_download_headers=add_docs_params.image_download_headers)
                )

        if add_docs_params.use_existing_tensors:
            doc_ids = []

            # Iterate through the list in reverse, only latest doc with dupe id gets added.
            for i in range(doc_count-1, -1, -1):
                if ("_id" in add_docs_params.docs[i]) and (add_docs_params.docs[i]["_id"] not in doc_ids):
                    doc_ids.append(add_docs_params.docs[i]["_id"])
            existing_docs = _get_documents_for_upsert(
                config=config, index_name=add_docs_params.index_name, document_ids=doc_ids)
        
        for i, doc in enumerate(add_docs_params.docs):

            indexing_instructions = {'index': {"_index": add_docs_params.index_name}}
            copied = copy.deepcopy(doc)

            document_is_valid = True
            new_fields_from_doc = set()

            doc_id = None
            try:
                validation.validate_doc(doc)

                if "_id" in doc:
                    doc_id = validation.validate_id(doc["_id"])
                    del copied["_id"]
                else:
                    doc_id = str(uuid.uuid4())

                [validation.validate_field_name(field) for field in copied]
            except errors.__InvalidRequestError as err:
                unsuccessful_docs.append(
                    (i, {'_id': doc_id if doc_id is not None else '',
                         'error': err.message, 'status': int(err.status_code), 'code': err.code})
                )
                continue

            indexing_instructions["index"]["_id"] = doc_id
            if add_docs_params.use_existing_tensors:
                matching_doc = [doc for doc in existing_docs["docs"] if doc["_id"] == doc_id]
                # Should only have 1 result, as only 1 id matches
                if len(matching_doc) == 1:
                    existing_doc = matching_doc[0]
                # When a request isn't sent to get matching docs, because the added docs don't
                # have IDs:
                elif len(matching_doc) == 0:
                    existing_doc = {"found": False}
                else:
                    raise errors.InternalError(message= f"Upsert: found {len(matching_doc)} matching docs for {doc_id} when only 1 or 0 should have been found.")

            doc_chunks = []
            for field in copied:
                # Validation phase for field
                try:
                    field_content = validation.validate_field_content(
                        field_content=copied[field],
                        is_non_tensor_field=not utils.is_tensor_field(field, add_docs_params.tensor_fields, add_docs_params.non_tensor_fields)
                    )
                    if isinstance(field_content, dict):
                        field_content = validation.validate_dict(
                            field=field, field_content=field_content,
                            is_non_tensor_field=not utils.is_tensor_field(field, add_docs_params.tensor_fields,
                                                                          add_docs_params.non_tensor_fields),
                            mappings=add_docs_params.mappings,
                            index_model_dimensions=index_model_dimensions)
                except errors.InvalidArgError as err:
                    document_is_valid = False
                    unsuccessful_docs.append(
                        (i, {'_id': doc_id, 'error': err.message, 'status': int(err.status_code),
                             'code': err.code})
                    )
                    break

                # Document field type used for:
                # 1. Determining how to add field to OpenSearch index
                # 2. Determining chunk and vectorise behavior
                # Multimodal fields not here because they will get added later (with nesting)
                document_field_type = add_docs.determine_document_field_type(field, field_content, add_docs_params.mappings)
                if field not in existing_fields:
                    if document_field_type == DocumentFieldType.standard:
                        # Normal field (str, int, float, bool, or list)
                        new_fields_from_doc.add((field, _infer_opensearch_data_type(field_content)))
                    elif document_field_type == DocumentFieldType.custom_vector:
                        # Custom vector field type in OpenSearch assimilates type of its content
                        new_fields_from_doc.add((field, _infer_opensearch_data_type(field_content["content"])))

                # Don't process text/image fields when explicitly told not to.
                if not utils.is_tensor_field(field, add_docs_params.tensor_fields, add_docs_params.non_tensor_fields):
                    continue

                # 4 current options for chunking/vectorisation behavior:
                # A) field type is custom_vector -> no chunking or vectorisation
                # B) use_existing_tensors=True and field content hasn't changed -> no chunking or vectorisation
                # C) field type is standard -> chunking and vectorisation
                # D) field type is multimodal -> use vectorise_multimodal_combination_field (does chunking and vectorisation)
                
                field_chunks_to_append = []     # Used by all cases. chunks generated by processing this field for this doc:
                
                # A) custom_vector fields -> no chunking or vectorisation
                if document_field_type == DocumentFieldType.custom_vector:
                    # No real vectorisation or chunking will happen for custom vector fields.
                    # No error handling, dict validation happens before this.
                    # Adding chunk (Only 1 chunk gets added for a custom vector field). No metadata yet.
                    field_chunks_to_append.append({
                        TensorField.marqo_knn_field: field_content["vector"],
                        TensorField.field_content: field_content["content"],
                        TensorField.field_name: field
                    })
                    # Update parent document (copied) to fit new format. Use content (text) to replace input dict
                    copied[field] = field_content["content"]

                # B) If field content hasn't changed -> use existing tensors, no chunking or vectorisation
                elif (add_docs_params.use_existing_tensors
                        and existing_doc["found"]
                        and (field in existing_doc["_source"]) and (existing_doc["_source"][field] == field_content)):
                    field_chunks_to_append = _get_chunks_for_field(field_name=field, doc_id=doc_id, doc=existing_doc)

                # Chunking and vectorising phase (only if content changed).
                # C) Standard document field type
                elif isinstance(field_content, (str, Image.Image)):

                    # TODO: better/consistent handling of a no-op for processing (but still vectorize)

                    # 1. check if urls should be downloaded -> "treat_pointers_and_urls_as_images":True
                    # 2. check if it is a url or pointer
                    # 3. If yes in 1 and 2, download blindly (without type)
                    # 4. Determine media type of downloaded
                    # 5. load correct media type into memory -> PIL (images), videos (), audio (torchaudio)
                    # 6. if chunking -> then add the extra chunker

                    if isinstance(field_content, str) and not _is_image(field_content):
                        # text processing pipeline:
                        split_by = index_info.index_settings[NsField.index_defaults][NsField.text_preprocessing][
                            NsField.split_method]
                        split_length = index_info.index_settings[NsField.index_defaults][NsField.text_preprocessing][
                            NsField.split_length]
                        split_overlap = index_info.index_settings[NsField.index_defaults][NsField.text_preprocessing][
                            NsField.split_overlap]
                        content_chunks = text_processor.split_text(field_content, split_by=split_by,
                                                                   split_length=split_length, split_overlap=split_overlap)
                        text_chunks = content_chunks
                    else:
                        # TODO put the logic for getting field parameters into a function and add per field options
                        image_method = index_info.index_settings[NsField.index_defaults][NsField.image_preprocessing][
                            NsField.patch_method]
                        # the chunk_image contains the no-op logic as of now - method = None will be a no-op
                        try:
                            # in the future, if we have different chunking methods, make sure we catch possible
                            # errors of different types generated here, too.
                            if isinstance(field_content, str) and index_info.index_settings[NsField.index_defaults][
                                NsField.treat_urls_and_pointers_as_images]:
                                if not isinstance(image_repo[field_content], Exception):
                                    image_data = image_repo[field_content]
                                else:
                                    raise s2_inference_errors.S2InferenceError(
                                        f"Could not find image found at `{field_content}`. \n"
                                        f"Reason: {str(image_repo[field_content])}"
                                    )
                            else:
                                image_data = field_content
                            if image_method not in [None, 'none', '', "None", ' ']:
                                content_chunks, text_chunks = image_processor.chunk_image(
                                    image_data, device=add_docs_params.device, method=image_method)
                            else:
                                # if we are not chunking, then we set the chunks as 1-len lists
                                # content_chunk is the PIL image
                                # text_chunk refers to URL
                                content_chunks, text_chunks = [image_data], [field_content]
                        except s2_inference_errors.S2InferenceError as e:
                            document_is_valid = False
                            unsuccessful_docs.append(
                                (i, {'_id': doc_id, 'error': e.message,
                                     'status': int(errors.InvalidArgError.status_code),
                                     'code': errors.InvalidArgError.code})
                            )
                            break

                    normalize_embeddings = index_info.index_settings[NsField.index_defaults][
                        NsField.normalize_embeddings]
                    infer_if_image = index_info.index_settings[NsField.index_defaults][
                        NsField.treat_urls_and_pointers_as_images]

                    try:
                        # in the future, if we have different underlying vectorising methods, make sure we catch possible
                        # errors of different types generated here, too.

                        # ADD DOCS TIMER-LOGGER (4)
                        start_time = timer()
                        with RequestMetricsStore.for_request().time(f"add_documents.create_vectors"):
                            vector_chunks = s2_inference.vectorise(
                                model_name=index_info.model_name,
                                model_properties=index_info.get_model_properties(), content=content_chunks,
                                device=add_docs_params.device, normalize_embeddings=normalize_embeddings,
                                infer=infer_if_image, model_auth=add_docs_params.model_auth
                            )

                        end_time = timer()
                        total_vectorise_time += (end_time - start_time)
                    except (s2_inference_errors.UnknownModelError,
                            s2_inference_errors.InvalidModelPropertiesError,
                            s2_inference_errors.ModelLoadError,
                            s2_inference.ModelDownloadError) as model_error:
                        raise errors.BadRequestError(
                            message=f'Problem vectorising query. Reason: {str(model_error)}',
                            link="https://marqo.pages.dev/latest/Models-Reference/dense_retrieval/"
                        )
                    except s2_inference_errors.S2InferenceError:
                        document_is_valid = False
                        image_err = errors.InvalidArgError(message=f'Could not process given image: {field_content}')
                        unsuccessful_docs.append(
                            (i, {'_id': doc_id, 'error': image_err.message, 'status': int(image_err.status_code),
                                 'code': image_err.code})
                        )
                        break
                    if (len(vector_chunks) != len(text_chunks)):
                        raise RuntimeError(
                            f"the input content after preprocessing and its vectorized counterparts must be the same length."
                            f"received text_chunks={len(text_chunks)} and vector_chunks={len(vector_chunks)}. "
                            f"check the preprocessing functions and try again. ")

                    for text_chunk, vector_chunk in zip(text_chunks, vector_chunks):
                        # Chunk added to field chunks_to_append (no metadata yet).
                        field_chunks_to_append.append({
                            TensorField.marqo_knn_field: vector_chunk,
                            TensorField.field_content: text_chunk,
                            TensorField.field_name: field
                        })

                # D) Multimodal chunking and vectorisation
                elif document_field_type == DocumentFieldType.multimodal_combination:
                    (combo_chunk, combo_document_is_valid,
                        unsuccessful_doc_to_append, combo_vectorise_time_to_add,
                        new_fields_from_multimodal_combination) = vectorise_multimodal_combination_field(
                            field, field_content, copied, i, doc_id, add_docs_params.device, index_info,
                            image_repo, add_docs_params.mappings[field], model_auth=add_docs_params.model_auth)
                    total_vectorise_time = total_vectorise_time + combo_vectorise_time_to_add
                    
                    if combo_document_is_valid is False:
                        unsuccessful_docs.append(unsuccessful_doc_to_append)
                        break
                    else:
                        if field not in new_obj_fields:
                            new_obj_fields[field] = set()
                        new_obj_fields[field] = new_obj_fields[field].union(new_fields_from_multimodal_combination)
                        # Multimodal combo chunk added to field_chunks_to_append (no metadata yet)
                        field_chunks_to_append.append(combo_chunk)

                # Executes REGARDLESS of document field type.
                # Add field_chunks_to_append to total document chunk list
                for chunk in field_chunks_to_append:
                    doc_chunks.append(chunk)

            if document_is_valid:
                new_fields = new_fields.union(new_fields_from_doc)

                # Create metadata to put in doc chunks (from altered doc)
                chunk_values_for_filtering = add_docs.create_chunk_metadata(raw_document=copied)
                for chunk in doc_chunks:
                    # Add metadata to each doc chunk
                    chunk.update(chunk_values_for_filtering)
                copied[TensorField.chunks] = doc_chunks

                bulk_parent_dicts.append(indexing_instructions)
                bulk_parent_dicts.append(copied)

        total_preproc_time = 0.001 * RequestMetricsStore.for_request().stop("add_documents.processing_before_opensearch")
        logger.debug(f"      add_documents pre-processing: took {(total_preproc_time):.3f}s total for {doc_count} docs, "
                    f"for an average of {(total_preproc_time / doc_count):.3f}s per doc.")

        logger.debug(f"          add_documents vectorise: took {(total_vectorise_time):.3f}s for {doc_count} docs, "
                    f"for an average of {(total_vectorise_time / doc_count):.3f}s per doc.")
        if bulk_parent_dicts:
            max_add_docs_retry_attempts = utils.read_env_vars_and_defaults(EnvVars.MARQO_OPENSEARCH_MAX_INDEX_RETRY_ATTEMPTS)
            max_add_docs_retry_backoff = utils.read_env_vars_and_defaults(EnvVars.MARQO_OPENSEARCH_MAX_INDEX_RETRY_BACKOFF)
            # the HttpRequest wrapper handles error logic
            update_mapping_response = backend.add_customer_field_properties(
                config=config, index_name=add_docs_params.index_name, customer_field_names=new_fields,
                multimodal_combination_fields=new_obj_fields, max_retry_attempts=max_add_docs_retry_attempts,
                max_retry_backoff_seconds=max_add_docs_retry_backoff)

            # ADD DOCS TIMER-LOGGER (5)
            start_time_5 = timer()
            with RequestMetricsStore.for_request().time("add_documents.opensearch._bulk"):
                serialised_body = utils.dicts_to_jsonl(bulk_parent_dicts)
                
                index_parent_response = HttpRequests(config).post(
                    path="_bulk", body=serialised_body,
                    max_retry_attempts=max_add_docs_retry_attempts,
                    max_add_docs_retry_backoff=max_add_docs_retry_backoff)
            RequestMetricsStore.for_request().add_time("add_documents.opensearch._bulk.internal", float(index_parent_response["took"]))

            end_time_5 = timer()
            total_http_time = end_time_5 - start_time_5
            total_index_time = index_parent_response["took"] * 0.001
            logger.debug(
                f"      add_documents roundtrip: took {(total_http_time):.3f}s to send {doc_count} docs (roundtrip) to Marqo-os, "
                f"for an average of {(total_http_time / doc_count):.3f}s per doc.")

            logger.debug(
                f"          add_documents Marqo-os index: took {(total_index_time):.3f}s for Marqo-os to index {doc_count} docs, "
                f"for an average of {(total_index_time / doc_count):.3f}s per doc.")
        else:
            index_parent_response = None

        with RequestMetricsStore.for_request().time("add_documents.postprocess"):
            if add_docs_params.auto_refresh:
                HttpRequests(config).post(path=F"{add_docs_params.index_name}/_refresh")

            t1 = timer()

            def translate_add_doc_response(response: Optional[dict], time_diff: float) -> dict:
                """translates OpenSearch response dict into Marqo dict"""
                item_fields_to_remove = ['_index', '_primary_term', '_seq_no', '_shards', '_version']
                result_dict = {}
                new_items = []

                if response is not None:
                    copied_res = copy.deepcopy(response)

                    result_dict['errors'] = copied_res['errors']
                    actioned = "index"

                    for item in copied_res["items"]:
                        for to_remove in item_fields_to_remove:
                            if to_remove in item[actioned]:
                                del item[actioned][to_remove]
                        new_items.append(item[actioned])

                if unsuccessful_docs:
                    result_dict['errors'] = True

                for loc, error_info in unsuccessful_docs:
                    new_items.insert(loc, error_info)

                result_dict["processingTimeMs"] = time_diff * 1000
                result_dict["index_name"] = add_docs_params.index_name
                result_dict["items"] = new_items
                return result_dict

            return translate_add_doc_response(response=index_parent_response, time_diff=t1 - t0)


def get_document_by_id(
        config: Config, index_name: str, document_id: str, show_vectors: bool = False):
    """returns document by its ID"""
    validation.validate_id(document_id)
    res = HttpRequests(config).get(
        f'{index_name}/_doc/{document_id}'
    )
    if "_source" in res:
        return _clean_doc(res["_source"], doc_id=document_id, include_vectors=show_vectors)
    else:
        return res


def get_documents_by_ids(
        config: Config, index_name: str, document_ids: List[str],
        show_vectors: bool = False,
):
    """returns documents by their IDs"""
    if not isinstance(document_ids, typing.Collection):
        raise errors.InvalidArgError("Get documents must be passed a collection of IDs!")
    if len(document_ids) <= 0:
        raise errors.InvalidArgError("Can't get empty collection of IDs!")
    max_docs_limit = utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_RETRIEVABLE_DOCS)
    if max_docs_limit is not None and len(document_ids) > int(max_docs_limit):
        raise errors.IllegalRequestedDocCount(
            f"{len(document_ids)} documents were requested, which is more than the allowed limit of [{max_docs_limit}], "
            f"set by the environment variable `{EnvVars.MARQO_MAX_RETRIEVABLE_DOCS}`")
    docs = [
        {"_index": index_name, "_id": validation.validate_id(doc_id)}
        for doc_id in document_ids
    ]
    if not show_vectors:
        for d in docs:
            d["_source"] = dict()
            d["_source"]["exclude"] = f"*{TensorField.vector_prefix}*"
    res = HttpRequests(config).get(
        f'_mget/',
        body={
            "docs": docs,
        }
    )
    if "docs" in res:
        to_return = {
            "results": []
        }
        for doc in res['docs']:
            if not doc['found']:
                to_return['results'].append({
                    '_id': doc['_id'],
                    TensorField.found: False})
            else:
                to_return['results'].append(
                    {TensorField.found: True,
                     **_clean_doc(doc["_source"], doc_id=doc["_id"], include_vectors=show_vectors)})
        return to_return
    else:
        return res


def _get_documents_for_upsert(
        config: Config, index_name: str, document_ids: List[str],
        show_vectors: bool = False,
):
    """returns document chunks and content"""
    if not isinstance(document_ids, typing.Collection):
        raise errors.InvalidArgError("Get documents must be passed a collection of IDs!")

    # If we receive an invalid ID, we skip it
    valid_doc_ids = []
    for d_id in document_ids:
        try:
            validation.validate_id(d_id)
            valid_doc_ids.append(d_id)
        except errors.InvalidDocumentIdError:
            pass

    if len(valid_doc_ids) <= 0:
        return {"docs": []}
    max_docs_limit = utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_RETRIEVABLE_DOCS)
    if max_docs_limit is not None and len(document_ids) > int(max_docs_limit):
        raise errors.IllegalRequestedDocCount(
            f"{len(document_ids)} documents were requested, which is more than the allowed limit of [{max_docs_limit}], "
            f"set by the environment variable `{EnvVars.MARQO_MAX_RETRIEVABLE_DOCS}`")

    # Chunk Docs (get field name, field content, vectors)

    chunk_docs = [
        {"_index": index_name, "_id": doc_id,
         "_source": {"include": [f"__chunks.__field_content", f"__chunks.__field_name", f"__chunks.__vector_*"]}}
        for doc_id in valid_doc_ids
    ]

    data_docs = [
        {"_index": index_name, "_id": doc_id, "_source": {"exclude": "__chunks.*"}}
        for doc_id in valid_doc_ids
    ]

    res = HttpRequests(config).get(
        f'_mget/',
        body={
            "docs": chunk_docs + data_docs,
        }
    )

    # Combine the 2 query results (loop through each doc id)
    combined_result = []

    for doc_id in valid_doc_ids:
        # There should always be 2 results per doc.
        result_list = [doc for doc in res["docs"] if doc["_id"] == doc_id]

        if len(result_list) == 0:
            continue
        if len(result_list) not in (2, 0):
            raise errors.InternalError(f"Internal error fetching old documents. "
                                       f"There are {len(result_list)} results for doc id {doc_id}.")

        for result in result_list:
            if result["found"]:
                doc_in_results = True
                if ("__chunks" in result["_source"]) and (result["_source"]["__chunks"] == []):
                    res_data = result
                else:
                    res_chunks = result
            else:
                doc_in_results = False
                dummy_res = result
                break

        # Put the chunks list in res_data, so it contains all doc data
        if doc_in_results:
            # Only add chunks if not a chunkless doc
            if res_chunks["_source"]:
                res_data["_source"]["__chunks"] = res_chunks["_source"]["__chunks"]
            combined_result.append(res_data)
        else:
            # This result just says that the doc was not found ("found": False)
            combined_result.append(dummy_res)

    res["docs"] = combined_result

    # Returns a list of combined docs
    return res


def refresh_index(config: Config, index_name: str):
    return HttpRequests(config).post(path=F"{index_name}/_refresh")


@add_timing
def bulk_search(query: BulkSearchQuery, marqo_config: config.Config, verbose: int = 0, device: str = None):
    """Performs a set of search operations in parallel.

    Args:
        query: Set of search queries
        marqo_config:
        verbose:
        device:

    Notes:
        Current limitations:
          - Lexical and tensor search done in serial.
          - A single error (e.g. validation errors) on any one of the search queries returns an error and does not
            process non-erroring queries.
    """
    refresh_indexes_in_background(marqo_config, [q.index for q in query.queries])

    # TODO: Let non-errored docs to propagate.
    errs = [validation.validate_bulk_query_input(q) for q in query.queries]
    if any(errs):
        err = next(e for e in errs if e is not None)
        raise err

    if len(query.queries) == 0:
        return {"result": []}

    if device is None:
        selected_device = utils.read_env_vars_and_defaults("MARQO_BEST_AVAILABLE_DEVICE")
        if selected_device is None:
            raise errors.InternalError("Best available device was not properly determined on Marqo startup.")
        logger.debug(f"No device given for bulk_search. Defaulting to best available device: {selected_device}")
    else:
        selected_device = device

    tensor_queries: Dict[int, BulkSearchQueryEntity] = dict(filter(lambda e: e[1].searchMethod == SearchMethod.TENSOR, enumerate(query.queries)))
    lexical_queries: Dict[int, BulkSearchQueryEntity] = dict(filter(lambda e: e[1].searchMethod == SearchMethod.LEXICAL, enumerate(query.queries)))

    tensor_search_results = dict(zip(tensor_queries.keys(), _bulk_vector_text_search(
            marqo_config, list(tensor_queries.values()), device=selected_device,
        )))

    # TODO: combine lexical + tensor queries into /_msearch
    lexical_search_results = dict(zip(lexical_queries.keys(), [_lexical_search(
        config=marqo_config, index_name=q.index, text=q.q, result_count=q.limit, offset=q.offset,
        searchable_attributes=q.searchableAttributes, verbose=verbose,
        filter_string=q.filter, attributes_to_retrieve=q.attributesToRetrieve
    ) for q in lexical_queries.values()]))

    # Recombine lexical and tensor in order
    combined_results = list({**tensor_search_results, **lexical_search_results}.items())
    combined_results.sort()
    search_results = [r[1] for r in combined_results]

    with RequestMetricsStore.for_request().time(f"bulk_search.rerank"):
        for i, s in enumerate(search_results):
            q = query.queries[i]
            s["query"] = q.q
            s["limit"] = q.limit
            s["offset"] = q.offset

            ## TODO: filter out highlights within `_lexical_search`
            if not q.showHighlights:
                for hit in s["hits"]:
                    del hit["_highlights"]

            if q.reRanker is not None:
                logger.debug(f"reranking {i}th query using {q.reRanker}")
                with RequestMetricsStore.for_request().time(f"bulk_search.{i}.rerank"):
                    rerank_query(q, s, q.reRanker, selected_device, 1)

    return {
        "result": search_results
    }


def rerank_query(query: BulkSearchQueryEntity, result: Dict[str, Any], reranker: Union[str, Dict], device: str, num_highlights: int):
    if query.searchableAttributes is None:
        raise errors.InvalidArgError(f"searchable_attributes cannot be None when re-ranking. Specify which fields to search and rerank over.")
    try:
        start_rerank_time = timer()
        rerank.rerank_search_results(search_result=result, query=query.q,
                                     model_name=reranker, device=device,
                                     searchable_attributes=query.searchableAttributes, num_highlights=num_highlights)
        logger.debug(f"search ({query.searchMethod.lower()}) reranking using {reranker}: took {(timer() - start_rerank_time):.3f}s to rerank results.")
    except Exception as e:
        raise errors.BadRequestError(f"reranking failure due to {str(e)}")


def refresh_indexes_in_background(config: Config, index_names: List[str]) -> None:
    """Refresh indices to index meta cache.
    """
    for idx in index_names:
        if idx not in index_meta_cache.get_cache():
            backend.get_index_info(config=config, index_name=idx)

        REFRESH_INTERVAL_SECONDS = 2
        # update cache in the background
        cache_update_thread = threading.Thread(
            target=index_meta_cache.refresh_index_info_on_interval,
            args=(config, idx, REFRESH_INTERVAL_SECONDS))
        cache_update_thread.start()


def search(config: Config, index_name: str, text: Optional[Union[str, dict]] = None,
           result_count: int = 3, offset: int = 0, highlights=True,
           search_method: Union[str, SearchMethod, None] = SearchMethod.TENSOR,
           searchable_attributes: Iterable[str] = None, verbose: int = 0,
           reranker: Union[str, Dict] = None, filter: str = None,
           attributes_to_retrieve: Optional[List[str]] = None,
           device: str = None, boost: Optional[Dict] = None,
           image_download_headers: Optional[Dict] = None,
           context: Optional[SearchContext] = None,
           score_modifiers: Optional[ScoreModifier] = None,
           model_auth: Optional[ModelAuth] = None) -> Dict:
    """The root search method. Calls the specific search method

    Validation should go here. Validations include:
        - all args and their types
        - result_count (negatives etc)
        - text

    This deals with index caching

    Args:
        config:
        index_name:
        text:
        result_count:
        offset:
        search_method:
        searchable_attributes:
        verbose:
        device: May be none, we calculate default device here
        num_highlights: number of highlights to return for each doc
        boost: boosters to re-weight the scores of individual fields
        image_download_headers: headers for downloading images
        context: a dictionary to allow custom vectors in search, for tensor search only
        score_modifiers: a dictionary to modify the score based on field values, for tensor search only
        model_auth: Authorisation details for downloading a model (if required)
    Returns:

    """

    # Validation for: result_count (limit) & offset
    # Validate neither is negative
    if result_count <= 0:
        raise errors.IllegalRequestedDocCount("search result limit must be greater than 0!")
    if offset < 0:
        raise errors.IllegalRequestedDocCount("search result offset cannot be less than 0!")

    validation.validate_query(q=text, search_method=search_method)

    # Validate result_count + offset <= int(max_docs_limit)
    max_docs_limit = utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_RETRIEVABLE_DOCS)
    check_upper = True if max_docs_limit is None else result_count + offset <= int(max_docs_limit)
    if not check_upper:
        upper_bound_explanation = ("The search result limit + offset must be less than or equal to the "
                                   f"MARQO_MAX_RETRIEVABLE_DOCS limit of [{max_docs_limit}]. ")

        raise errors.IllegalRequestedDocCount(
            f"{upper_bound_explanation} Marqo received search result limit of `{result_count}` "
            f"and offset of `{offset}`.")

    t0 = timer()
    validation.validate_context(context=context, query=text, search_method=search_method)
    validation.validate_boost(boost=boost, search_method=search_method)
    validation.validate_searchable_attributes(searchable_attributes=searchable_attributes, search_method=search_method)
    if searchable_attributes is not None:
        [validation.validate_field_name(attribute) for attribute in searchable_attributes]
    if attributes_to_retrieve is not None:
        if not isinstance(attributes_to_retrieve, (List, typing.Tuple)):
            raise errors.InvalidArgError("attributes_to_retrieve must be a sequence!")
        [validation.validate_field_name(attribute) for attribute in attributes_to_retrieve]
    if verbose:
        print(f"determined_search_method: {search_method}, text query: {text}")
    # if we can't see the index name in cache, we request it and wait for the info
    if index_name not in index_meta_cache.get_cache():
        backend.get_index_info(config=config, index_name=index_name)

    REFRESH_INTERVAL_SECONDS = 2
    # update cache in the background
    cache_update_thread = threading.Thread(
        target=index_meta_cache.refresh_index_info_on_interval,
        args=(config, index_name, REFRESH_INTERVAL_SECONDS))
    cache_update_thread.start()

    if device is None:
        selected_device = utils.read_env_vars_and_defaults("MARQO_BEST_AVAILABLE_DEVICE")
        if selected_device is None:
            raise errors.InternalError("Best available device was not properly determined on Marqo startup.")
        logger.debug(f"No device given for search. Defaulting to best available device: {selected_device}")
    else:
        selected_device = device

    if search_method.upper() == SearchMethod.TENSOR:
        search_result = _vector_text_search(
            config=config, index_name=index_name, query=text, result_count=result_count, offset=offset,
            searchable_attributes=searchable_attributes, verbose=verbose,
            filter_string=filter, device=selected_device, attributes_to_retrieve=attributes_to_retrieve, boost=boost,
            image_download_headers=image_download_headers, context=context, score_modifiers=score_modifiers,
            model_auth=model_auth
        )
    elif search_method.upper() == SearchMethod.LEXICAL:
        search_result = _lexical_search(
            config=config, index_name=index_name, text=text, result_count=result_count, offset=offset,
            searchable_attributes=searchable_attributes, verbose=verbose,
            filter_string=filter, attributes_to_retrieve=attributes_to_retrieve
        )
    else:
        raise errors.InvalidArgError(f"Search called with unknown search method: {search_method}")

    if reranker is not None:
        logger.info("reranking using {}".format(reranker))
        if searchable_attributes is None:
            raise errors.InvalidArgError(
                f"searchable_attributes cannot be None when re-ranking. Specify which fields to search and rerank over.")
        try:
            # SEARCH TIMER-LOGGER (reranking)
            RequestMetricsStore.for_request().start(f"search.rerank")
            rerank.rerank_search_results(search_result=search_result, query=text,
                                         model_name=reranker,
                                         device=selected_device,
                                         searchable_attributes=searchable_attributes,
                                         num_highlights=1)
            total_rerank_time = RequestMetricsStore.for_request().stop(f"search.rerank")
            logger.debug(
                f"search ({search_method.lower()}) reranking using {reranker}: took {(total_rerank_time):.3f}ms to rerank results."
            )
        except Exception as e:
            raise errors.BadRequestError(f"reranking failure due to {str(e)}")

    search_result["query"] = text
    search_result["limit"] = result_count
    search_result["offset"] = offset

    if not highlights:
        for hit in search_result["hits"]:
            del hit["_highlights"]

    time_taken = timer() - t0
    search_result["processingTimeMs"] = round(time_taken * 1000)
    logger.debug(f"search ({search_method.lower()}) completed with total processing time: {(time_taken):.3f}s.")

    return search_result


def _lexical_search(
        config: Config, index_name: str, text: str, result_count: int = 3, offset: int = 0,
        searchable_attributes: Sequence[str] = None, verbose: int = 0, filter_string: str = None,
        attributes_to_retrieve: Optional[List[str]] = None, expose_facets: bool = False):
    """

    Args:
        config:
        index_name:
        text:
        result_count:
        offset:
        searchable_attributes:
        verbose:

    Returns:

    Notes:
        Should not be directly called by client - the search() method should
        be called. The search() method adds syncing
        Uses normal search (not multiple search).
    TODO:
        - Test raise_for_searchable_attribute=False
    """
    if not isinstance(text, str):
        raise errors.InvalidArgError(
            f"Lexical search query arg must be of type `str`! text arg is of type {type(text)}. "
            f"Query arg: {text}")

    # SEARCH TIMER-LOGGER (pre-processing)
    RequestMetricsStore.for_request().start("search.lexical.processing_before_opensearch")
    if searchable_attributes is not None:
        # Empty searchable attributes should produce empty results.
        if len(searchable_attributes) == 0:
            return {"hits": []}

        fields_to_search = searchable_attributes
    else:
        fields_to_search = index_meta_cache.get_index_info(
            config=config, index_name=index_name
        ).get_true_text_properties()

    # Parse text into required and optional terms.
    (required_terms, optional_blob) = utils.parse_lexical_query(text)

    body = {
        "query": {
            "bool": {
                # Optional blob terms SHOULD be in results.
                "should": [
                    {"match": {field: optional_blob}}
                    for field in fields_to_search
                ],
                # Required terms MUST be in results.
                "must": [
                    {
                        # Nested bool, since required term can be in ANY field.
                        "bool": {
                            "should": [
                                {"match_phrase": {field: term}}
                                for field in fields_to_search
                            ]
                        }
                    }
                    for term in required_terms
                ],
                "must_not": [
                    {"exists": {"field": TensorField.field_name}}
                ],
            }
        },
        "size": result_count,
        "from": offset
    }

    if filter_string is not None:
        body["query"]["bool"]["filter"] = [{
            "query_string": {"query": filter_string}}]
    if attributes_to_retrieve is not None:
        body["_source"] = {"include": attributes_to_retrieve} if len(attributes_to_retrieve) > 0 else False
    if not expose_facets:
        if "_source" not in body:
            body["_source"] = dict()
        if body["_source"] is not False:
            body["_source"]["exclude"] = [f"*{TensorField.vector_prefix}*"]

    total_preprocess_time = RequestMetricsStore.for_request().stop("search.lexical.processing_before_opensearch")
    logger.debug(f"search (lexical) pre-processing: took {(total_preprocess_time):.3f}ms to process query.")

    max_search_retry_attempts = utils.read_env_vars_and_defaults(EnvVars.MARQO_OPENSEARCH_MAX_SEARCH_RETRY_ATTEMPTS)
    max_search_retry_backoff = utils.read_env_vars_and_defaults(EnvVars.MARQO_OPENSEARCH_MAX_SEARCH_RETRY_BACKOFF)
    start_search_http_time = timer()
    with RequestMetricsStore.for_request().time("search.opensearch._search"):
        search_res = HttpRequests(config).get(
            path=f"{index_name}/_search",
            body=body,
            max_retry_attempts=max_search_retry_attempts,
            max_retry_backoff_seconds=max_search_retry_backoff
        )
    RequestMetricsStore.for_request().add_time("search.opensearch._search.internal", search_res["took"] * 0.001) # internal, not round trip time

    end_search_http_time = timer()
    total_search_http_time = end_search_http_time - start_search_http_time
    total_os_process_time = search_res["took"] * 0.001
    num_results = len(search_res['hits']['hits'])
    logger.debug(
        f"search (lexical) roundtrip: took {(total_search_http_time):.3f}s to send search query (roundtrip) to Marqo-os and received {num_results} results.")
    logger.debug(
        f"  search (lexical) Marqo-os processing time: took {(total_os_process_time):.3f}s for Marqo-os to execute the search.")

    # SEARCH TIMER-LOGGER (post-processing)
    RequestMetricsStore.for_request().start("search.lexical.postprocess")
    res_list = []
    for doc in search_res['hits']['hits']:
        just_doc = _clean_doc(doc["_source"].copy()) if "_source" in doc else dict()
        just_doc["_id"] = doc["_id"]
        just_doc["_score"] = doc["_score"]
        res_list.append({**just_doc, "_highlights": []})

    total_postprocess_time = RequestMetricsStore.for_request().stop("search.lexical.postprocess")
    logger.debug(
        f"search (lexical) post-processing: took {(total_postprocess_time):.3f}ms to format {len(res_list)} results.")

    return {'hits': res_list}


def construct_vector_input_batches(query: Union[str, Dict, None], index_info: IndexInfo) -> Tuple[List[str], List[str]]:
    """Splits images from text in a single query (either a query string, or dict of weighted strings).

    Args:
        query: a string query, or a dict of weighted strings.
        index_info: used to determine whether URLs should be treated as images

    Returns:
        A tuple of string batches. The first is text content the second is image content.
    """
    treat_urls_as_images = index_info.index_settings[NsField.index_defaults][NsField.treat_urls_and_pointers_as_images]
    if query is None:
        # Nothing should be vectorised.
        return [], []
    elif isinstance(query, str):
        if treat_urls_as_images and _is_image(query):
            return [], [query, ]
        else:
            return [query, ], []
    else:  # is dict:
        ordered_queries = list(query.items())
        if treat_urls_as_images:
            text_queries = [k for k, _ in ordered_queries if not _is_image(k)]
            image_queries = [k for k, _ in ordered_queries if _is_image(k)]
            return text_queries, image_queries
        else:
            return [k for k, _ in ordered_queries], []


def construct_msearch_body_elements(searchableAttributes: List[str], offset: int, filter_string: str,
                                    index_info: IndexInfo, result_count: int, query_vector: List[float],
                                    attributes_to_retrieve: List[str], index_name: str,
                                    score_modifiers: Optional[ScoreModifier] = None) -> List[Dict[str, Any]]:
    """Constructs the body payload of a `/_msearch` request for a single bulk search query"""

    # search filter has 2 components:
    # 1. searchable_attributes filter: filters out results that are not part of the searchable attributes
    # 2. filter_string: filters out results that do not match the filter string

    body = []

    if not utils.check_is_zero_vector(query_vector):
        filter_for_opensearch = filtering.build_tensor_search_filter(
            filter_string=filter_string, simple_properties=index_info.get_text_properties(),
            searchable_attribs=searchableAttributes
        )
        if score_modifiers is not None:
            search_query = _create_score_modifiers_tensor_search_query(
                score_modifiers,
                result_count,
                offset,
                TensorField.marqo_knn_field,
                query_vector
            )
            if (filter_string is not None) or (searchableAttributes is not None):
                (search_query["query"]["function_score"]
                    ["query"]["nested"]
                    ["query"]["function_score"]
                    ["query"]["knn"]
                    [f"{TensorField.chunks}.{TensorField.marqo_knn_field}"]["filter"]) = {
                        "query_string": {"query": f"{filter_for_opensearch}"}
                    }
        else:
            search_query = _create_normal_tensor_search_query(result_count, offset, TensorField.marqo_knn_field, query_vector)
            if (filter_string is not None) or (searchableAttributes is not None):
                (search_query["query"]["nested"]
                    ["query"]["knn"]
                    [f"{TensorField.chunks}.{TensorField.marqo_knn_field}"]["filter"]) = {
                        "query_string": {"query": f"{filter_for_opensearch}"}
                    }

        if attributes_to_retrieve is not None:
            search_query["_source"] = {"include": attributes_to_retrieve} if len(attributes_to_retrieve) > 0 else False
    else:
        search_query = _create_dummy_query_for_zero_vector_search()

    body += [{"index": index_name}, search_query]

    return body


def bulk_msearch(config: Config, body: List[Dict]) -> List[Dict]:
    """Send an `/_msearch` request to MarqoOS and translate errors into a user-friendly format."""
    max_search_retry_attempts = utils.read_env_vars_and_defaults(EnvVars.MARQO_OPENSEARCH_MAX_SEARCH_RETRY_ATTEMPTS)
    max_search_retry_backoff = utils.read_env_vars_and_defaults(EnvVars.MARQO_OPENSEARCH_MAX_SEARCH_RETRY_BACKOFF)
    start_search_http_time = timer()
    try:
        with RequestMetricsStore.for_request().time("search.opensearch._msearch"):
            serialised_search_body = utils.dicts_to_jsonl(body)
            response = HttpRequests(config).get(
                path=F"_msearch",
                body=serialised_search_body,
                max_retry_attempts=max_search_retry_attempts,
                max_retry_backoff_seconds=max_search_retry_backoff
            )
        RequestMetricsStore.for_request().add_time("search.opensearch._msearch.internal", float(response["took"])) # internal, not round trip time

        end_search_http_time = timer()
        total_search_http_time = end_search_http_time - start_search_http_time
        total_os_process_time = response["took"] * 0.001
        num_responses = len(response["responses"])
        logger.debug(f"search (tensor) roundtrip: took {total_search_http_time:.3f}s to send {num_responses} search queries (roundtrip) to Marqo-os.")

        responses = [r['hits']['hits'] for r in response["responses"]]

    except KeyError as e:
        # KeyError indicates we have received a non-successful result
        try:
            root_cause_reason = response["responses"][0]["error"]["root_cause"][0]["reason"]
            root_cause_type: Optional[str] = response["responses"][0]["error"]["root_cause"][0].get("type")

            if "index.max_result_window" in root_cause_reason:
                raise errors.IllegalRequestedDocCount("Marqo-OS rejected the response due to too many requested results. Try reducing the query's limit parameter") from e
            elif 'parse_exception' in root_cause_reason:
                raise errors.InvalidArgError("Syntax error, could not parse filter string") from e
            elif root_cause_type == 'query_shard_exception':
                if root_cause_reason.startswith("Failed to parse query"):
                    raise errors.InvalidArgError("Syntax error, could not parse filter string") from e
                elif "Query vector has invalid dimension" in root_cause_reason:
                    raise errors.InvalidArgError("Tensor search query or context vector given has invalid dimensions. Please check your index dimensions before searching again") from e
            raise errors.BackendCommunicationError(f"Error communicating with Marqo-OS backend:\n{response}")
        except (KeyError, IndexError):
            raise e

    logger.debug(f"  search (tensor) Marqo-os processing time: took {total_os_process_time:.3f}s for Marqo-os to execute the search.")
    return responses

def gather_documents_from_response(resp: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
        For the specific responses to a query, gather correct responses. This is used to aggregate documents, for cases
        where the same document is retrieved for multiple field-queries.
    """
    gathered_docs = dict()

    for i, query_res in enumerate(resp):
        for doc in query_res:
            doc_chunks = doc["inner_hits"][TensorField.chunks]["hits"]["hits"]
            if doc["_id"] in gathered_docs:
                gathered_docs[doc["_id"]]["doc"] = doc
                gathered_docs[doc["_id"]]["chunks"].extend(doc_chunks)
            else:
                gathered_docs[doc["_id"]] = {
                    "_id": doc["_id"],
                    "doc": doc,
                    "chunks": doc_chunks
                }

        # Filter out docs with no inner hits:
        for doc_id in list(gathered_docs.keys()):
            if not gathered_docs[doc_id]["chunks"]:
                del gathered_docs[doc_id]

    return gathered_docs


def assign_query_to_vector_job(
        q: BulkSearchQueryEntity, jobs: Dict[JHash, VectorisedJobs], grouped_content: Tuple[List[str], List[str]],
        index_info: IndexInfo, device: str) -> List[VectorisedJobPointer]:
    """
    For a individual query, assign its content (to be vectorised) to a vector job. If none exist with the correct
    specifications, create a new job.

    Mutates entries in, and adds values to, the `jobs` param.

    Args:
        q:
        jobs:
        grouped_content: a 2-tuple of content, belonging to a single query, the first element is a list of text content.
            The second is a list of image URLs. Either element can be an empty list
        index_info:
        device:

    Returns:
        A list of pointers to the location in a vector job that will have its vectorised content.
    """
    if len(grouped_content) != 2:
        raise RuntimeError(
            "assign_query_to_vector_job() expects param `grouped_content` with 2 elems. Instead received"
            f" `grouped_content` with {len(grouped_content)} elems")
    ptrs = []
    for i, grouped_content in enumerate(grouped_content):
        content_type = 'text' if i == 0 else 'image'
        vector_job = VectorisedJobs(
            model_name=index_info.model_name,
            model_properties=index_info.get_model_properties(),
            content=grouped_content,
            device=device,
            normalize_embeddings=index_info.index_settings['index_defaults']['normalize_embeddings'],
            image_download_headers=q.image_download_headers,
            content_type=content_type,
            model_auth=q.modelAuth
        )
        # If exists, add content to vector job. Otherwise create new
        if jobs.get(vector_job.groupby_key()) is not None:
            j = jobs.get(vector_job.groupby_key())
            ptrs.append(j.add_content(grouped_content))
        else:
            jobs[vector_job.groupby_key()] = vector_job
            ptrs.append(VectorisedJobPointer(
                job_hash=vector_job.groupby_key(),
                start_idx=0,
                end_idx=len(vector_job.content)
            ))
    return ptrs


def create_vector_jobs(queries: List[BulkSearchQueryEntity], config: Config, device: str) -> Tuple[Dict[Qidx, List[VectorisedJobPointer]], Dict[JHash, VectorisedJobs]]:
    """
        For each query:
            - Find what needs to be vectorised
            - Group content (across search requests), that could be vectorised together
            - Keep track of the Job related to a search query

        Returns:
            - A mapping of the query index to the VectorisedJobPointer that points to the VectorisedJobs that will process its content.
            - A mapping of job key to job (for fast access).
    """
    qidx_to_job: Dict[Qidx, List[VectorisedJobPointer]] = dict()
    jobs: Dict[JHash, VectorisedJobs] = {}
    for i, q in enumerate(queries):
        q = queries[i]
        index_info = get_index_info(config=config, index_name=q.index)
        # split images from text:
        to_be_vectorised: Tuple[List[str], List[str]] = construct_vector_input_batches(q.q, index_info)
        qidx_to_job[i] = assign_query_to_vector_job(q, jobs, to_be_vectorised, index_info, device)

    return qidx_to_job, jobs


def vectorise_jobs(jobs: List[VectorisedJobs]) -> Dict[JHash, Dict[str, List[float]]]:
    """ Run s2_+inference.vectorise() on against each vector jobs.
    TODO: return a mapping of mapping: <JHash: <content: vector> >
    """
    result: Dict[JHash, Dict[str, List[float]]] = dict()
    for v in jobs:
        # TODO: Handle exception for single job, and allow others to run.
        try:
            if v.content:
                vectors = s2_inference.vectorise(
                    model_name=v.model_name, model_properties=v.model_properties,
                    content=v.content, device=v.device,
                    normalize_embeddings=v.normalize_embeddings,
                    image_download_headers=v.image_download_headers,
                    model_auth=v.model_auth
                )
                result[v.groupby_key()] = dict(zip(v.content, vectors))

        # TODO: This is a temporary addition.
        except (s2_inference_errors.UnknownModelError,
                s2_inference_errors.InvalidModelPropertiesError,
                s2_inference_errors.ModelLoadError,
                s2_inference.ModelDownloadError) as model_error:
            raise errors.BadRequestError(
                message=f'Problem vectorising query. Reason: {str(model_error)}',
                link="https://marqo.pages.dev/latest/Models-Reference/dense_retrieval/"
            )

        except s2_inference_errors.S2InferenceError as e:
            # TODO: differentiate image processing errors from other types of vectorise errors
            raise errors.InvalidArgError(message=f'Error vectorising content: {v.content}. Message: {e}')
    return result


def get_query_vectors_from_jobs(
        queries: List[BulkSearchQueryEntity], qidx_to_job: Dict[Qidx, List[VectorisedJobPointer]],
        job_to_vectors: Dict[JHash, Dict[str, List[float]]], config: Config,
        jobs: Dict[JHash, VectorisedJobs]
) -> Dict[Qidx, List[float]]:
    """
    Retrieve the vectorised content associated to each query from the set of batch vectorise jobs.
    Handles multi-modal queries, by weighting and combining queries into a single vector

    Args:
        - queries: Original search queries.
        - qidx_to_job: VectorisedJobPointer for each query
        - job_to_vectors: inference output from each VectorisedJob
        - config: standard Marqo config.

    """
    result: Dict[Qidx, List[float]] = defaultdict(list)
    for qidx, ptrs in qidx_to_job.items():
        q = queries[qidx]
        index_info = get_index_info(config=config, index_name=q.index)
        # Normal single text query
        if isinstance(q.q, str):
            result[qidx] = get_content_vector(
                possible_jobs=qidx_to_job.get(qidx, []),
                jobs=jobs,
                job_to_vectors= job_to_vectors,
                treat_urls_as_images=index_info.index_settings[NsField.index_defaults][NsField.treat_urls_and_pointers_as_images],
                content=q.q
            )

        elif isinstance(q.q, dict) or q.q is None:
            # Collect and weight all context tensors
            context_tensors = q.get_context_tensor()
            if context_tensors is not None:
                weighted_context_vectors = [np.asarray(v.vector) * v.weight for v in context_tensors]
            else:
                weighted_context_vectors = []
            # No query
            if q.q is None:
                weighted_vectors = weighted_context_vectors
            else:
                # Multiple queries. We have to weight and combine them:
                ordered_queries = list(q.q.items()) if isinstance(q.q, dict) else None
                if ordered_queries:
                    vectorised_ordered_queries = [
                        (get_content_vector(
                            possible_jobs=qidx_to_job[qidx],
                            jobs=jobs,
                            job_to_vectors=job_to_vectors,
                            treat_urls_as_images=index_info.index_settings[NsField.index_defaults][NsField.treat_urls_and_pointers_as_images],
                            content=content),
                        weight,
                        content
                        ) for content, weight in ordered_queries
                    ]
                    # TODO how do we ensure order?
                    weighted_query_vectors = [np.asarray(vec) * weight for vec, weight, content in vectorised_ordered_queries]

                    # Combine query and context vectors
                    weighted_vectors = weighted_query_vectors + weighted_context_vectors

            # Merge and normalize (if needed) all vectors
            try:
                merged_vector = np.mean(weighted_vectors, axis=0)
            except ValueError as e:
                raise errors.InvalidArgError(f"The provided vectors are not in the same dimension of the index."
                                            f"This causes the error when we do `numpy.mean()` over all the vectors.\n"
                                            f"The original error is `{e}`.\n"
                                            f"Please check `https://docs.marqo.ai/0.0.16/API-Reference/search/#context`.")
            if index_info.index_settings['index_defaults']['normalize_embeddings']:
                norm = np.linalg.norm(merged_vector, axis=-1, keepdims=True)
                if norm > 0:
                    # TODO: Why are we calculating norm twice?
                    merged_vector /= np.linalg.norm(merged_vector, axis=-1, keepdims=True)
            result[qidx] = list(merged_vector)
        
        else:
            raise errors.InternalError(f"Query can only be `str`, `dict`, or `None`. Invalid query type received: {type(q.q)}")

    return result


def get_content_vector(possible_jobs: List[VectorisedJobPointer], job_to_vectors: Dict[JHash, Dict[str, List[float]]],
                       jobs: Dict[JHash, VectorisedJobs],
                       treat_urls_as_images: bool, content: str) -> List[float]:
    """finds the vector associated with a piece of content

    Args:
        possible_jobs: The jobs where the target vector may reside
        treat_urls_as_images: an index_parameter that indicates whether content should be treated as image,
            if it has a URL structure
        content: The content to search

    Returns:
        Associated vector, if it is found.

    Raises runtime error if is not found
    """
    content_type = 'image' if treat_urls_as_images and _is_image(content) else 'text'
    not_found_error = RuntimeError(f"get_content_vector(): could not find corresponding vector for content `{content}`")
    for vec_job_pointer in possible_jobs:
        if jobs[vec_job_pointer.job_hash].content_type == content_type:
            try:
                return job_to_vectors[vec_job_pointer.job_hash][content]
            except KeyError:
                raise not_found_error
    raise not_found_error


def create_empty_query_response(queries: List[BulkSearchQueryEntity]) -> List[Dict]:
    return list(
        map(
            lambda x: {"hits": []}, queries
        )
    )

def run_vectorise_pipeline(config: Config, queries: List[BulkSearchQueryEntity], device: Union[Device, str]) -> Dict[Qidx, List[float]]:
    """Run the query vectorisation process"""
    # 1. Pre-process inputs ready for s2_inference.vectorise
    # we can still use qidx_to_job. But the jobs structure may need to be different
    vector_jobs_tuple: Tuple[Dict[Qidx, List[VectorisedJobPointer]], Dict[JHash, VectorisedJobs]] = create_vector_jobs(queries, config, device)

    qidx_to_jobs, jobs = vector_jobs_tuple

    # 2. Vectorise in batches against all queries
    ## TODO: To ensure that we are vectorising in batches, we can mock vectorise (), and see if the number of calls is as expected (if batch_size = 16, and number of docs = 32, and all args are the same, then number of calls = 2)
    # TODO: we need to enable str/PIL image structure:
    job_ptr_to_vectors: Dict[JHash, Dict[str, List[float]]] = vectorise_jobs(list(jobs.values()))

    # 3. For each query, get associated vectors
    qidx_to_vectors: Dict[Qidx, List[float]] = get_query_vectors_from_jobs(
        queries, qidx_to_jobs, job_ptr_to_vectors, config, jobs
    )
    return qidx_to_vectors

def _bulk_vector_text_search(config: Config, queries: List[BulkSearchQueryEntity], device: str = None) -> List[Dict]:
    """Resolve a batch of search queries in parallel.

    Args:
        - config:
        - queries: A list of independent search queries. Can be across multiple indexes, but are all expected to have `searchMethod = "TENSOR"`
    Returns:
        A list of search query responses (see `_format_ordered_docs_simple` for structure of individual entities).
    Note:
        - Search results are in the same order as `queries`.
        - device should ALWAYS be set, because it is only called by _bulk_search with the parameter specified
    """

    if len(queries) == 0:
        return []

    if not device:
        raise errors.InternalError("_bulk_vector_text_search cannot be called without `device`!")

    with RequestMetricsStore.for_request().time("bulk_search.vector.processing_before_opensearch",
        lambda t : logger.debug(f"bulk search (tensor) pre-processing: took {t:.3f}ms")
    ):

        with RequestMetricsStore.for_request().time(f"bulk_search.vector_inference_full_pipeline"):
            qidx_to_vectors: Dict[Qidx, List[float]] = run_vectorise_pipeline(config, queries, device)

        ## 4. Create msearch request bodies and combine to aggregate.
        query_to_body_parts: Dict[Qidx, List[Dict]] = dict()
        query_to_body_count: Dict[Qidx, int] = dict() # Keep track of count, so we can separate after msearch call.
        for qidx, q in enumerate(queries):
            index_info = get_index_info(config=config, index_name=q.index)
            body = construct_msearch_body_elements(q.searchableAttributes, q.offset, q.filter, index_info, q.limit, qidx_to_vectors[qidx], q.attributesToRetrieve, q.index, q.scoreModifiers)
            query_to_body_parts[qidx] = body
            query_to_body_count[qidx] = len(body)

        # Combine all msearch request bodies into one request body.
        aggregate_body = functools.reduce(lambda x, y: x + y, query_to_body_parts.values())
        if not aggregate_body:
            # Must return empty response, per search query
            return create_empty_query_response(queries)

    ## 5. POST aggregate  to /_msearch
    responses = bulk_msearch(config, aggregate_body)

    with RequestMetricsStore.for_request().time("bulk_search.vector.postprocess",
        lambda t : logger.debug(f"bulk search (tensor) post-processing: took {t:.3f}ms")
    ):
        # 6. Get documents back to each query, perform "gather" operation
        return create_bulk_search_response(queries, query_to_body_count, responses)


def create_bulk_search_response(queries: List[BulkSearchQueryEntity], query_to_body_count: Dict[Qidx, int], responses) -> List[Dict]:
    """
        Create Marqo search responses by extracting the appropriate elements from the batched /_msearch response. Also handles:
            - Boosting score (optional)
            - Sorting chunks
            - Formatting style
            - (no) highlights
        Does not mutate `responses` param.

    """
    results = []
    msearch_resp = copy.deepcopy(responses)
    for qidx, count in query_to_body_count.items():
        num_of_docs = count // 2
        result = msearch_resp[:num_of_docs]
        msearch_resp = msearch_resp[num_of_docs:]  # remove docs from response for next query

        query = queries[qidx]
        gathered_docs = gather_documents_from_response(result)
        if query.boost is not None:
            gathered_docs = boost_score(gathered_docs, query.boost, query.searchableAttributes)
        docs_chunks_sorted = sort_chunks(gathered_docs)
        results.append(
            _format_ordered_docs_simple(ordered_docs_w_chunks=docs_chunks_sorted, result_count=query.limit)
        )

    return results

def _vector_text_search(
        config: Config, index_name: str, query: Union[str, dict, None], result_count: int = 5, offset: int = 0,
        searchable_attributes: Iterable[str] = None, verbose=0, filter_string: str = None, device: str = None,
        attributes_to_retrieve: Optional[List[str]] = None, boost: Optional[Dict] = None,
        image_download_headers: Optional[Dict] = None, context: Optional[Dict] = None,
        score_modifiers: Optional[ScoreModifier] = None, model_auth: Optional[ModelAuth] = None):
    """

    Args:
        config:
        index_name:
        query: either a string query (which can be a URL or natural language text), or a dict of
            <query string>:<weight float> pairs.
        result_count:
        offset:
        searchable_attributes: Iterable of field names to search. If left as None, then all will
            be searched
        verbose: if 0 - nothing is printed. if 1 - data is printed without vectors, if 2 - full
            objects are printed out
        attributes_to_retrieve: if set, only returns these fields
        image_download_headers: headers for downloading images
        context: a dictionary to allow custom vectors in search
        score_modifiers: a dictionary to modify the score based on field values, for tensor search only
        model_auth: Authorisation details for downloading a model (if required)
    Returns:

    Note:
        - uses multisearch, which returns k results in each attribute. Not that much of a concern unless you have a
        ridiculous number of attributes
        - Should not be directly called by client - the search() method should
        be called. The search() method adds syncing
        - device should ALWAYS be set

    Output format:
        [
            {
                _id: doc_id
                doc: {# original document},
                highlights:[{}],
            },
        ]
    Future work:
        - max result count should be in a config somewhere
        - searching a non existent index should return a HTTP-type error
    """
    # # SEARCH TIMER-LOGGER (pre-processing)
    if not device:
        raise errors.InternalError("_vector_text_search cannot be called without `device`!")

    RequestMetricsStore.for_request().start("search.vector.processing_before_opensearch")

    try:
        index_info = get_index_info(config=config, index_name=index_name)
    except KeyError as e:
        raise errors.IndexNotFoundError(message="Tried to search a non-existent index: {}".format(index_name))

    queries = [BulkSearchQueryEntity(
        q=query, searchableAttributes=searchable_attributes,searchMethod=SearchMethod.TENSOR, limit=result_count, offset=offset, showHighlights=False, filter=filter_string, attributesToRetrieve=attributes_to_retrieve, boost=boost, image_download_headers=image_download_headers, context=context, scoreModifiers=score_modifiers, index=index_name, modelAuth=model_auth
    )]
    with RequestMetricsStore.for_request().time(f"search.vector_inference_full_pipeline"):
        qidx_to_vectors: Dict[Qidx, List[float]] = run_vectorise_pipeline(config, queries, device)
    vectorised_text = list(qidx_to_vectors.values())[0]

    body = construct_msearch_body_elements(
        searchable_attributes, offset, filter_string, index_info, result_count, vectorised_text, attributes_to_retrieve, index_name, score_modifiers
    )

    if verbose:
        _vector_text_search_query_verbose(verbose=verbose, body=body)

    total_preprocess_time = RequestMetricsStore.for_request().stop("search.vector.processing_before_opensearch")
    logger.debug(f"search (tensor) pre-processing: took {(total_preprocess_time):.3f}ms to vectorize and process query.")

    # SEARCH TIMER-LOGGER (roundtrip)
    responses = bulk_msearch(config, body)

    # SEARCH TIMER-LOGGER (post-processing)
    RequestMetricsStore.for_request().start("search.vector.postprocess")
    gathered_docs = gather_documents_from_response(responses)

    if boost is not None:
        gathered_docs = boost_score(gathered_docs, boost, searchable_attributes)

    completely_sorted = sort_chunks(gathered_docs)

    if verbose:
        print("Chunk vector search, sorted result:")
        if verbose == 1:
            pprint.pprint(utils.truncate_dict_vectors(completely_sorted))
        elif verbose == 2:
            pprint.pprint(completely_sorted)

    res = _format_ordered_docs_simple(ordered_docs_w_chunks=completely_sorted, result_count=result_count)

    total_postprocess_time = RequestMetricsStore.for_request().stop("search.vector.postprocess")
    logger.debug(
        f"search (tensor) post-processing: took {(total_postprocess_time):.3f}ms to sort and format {len(completely_sorted)} results from Marqo-os.")
    return res

def _format_ordered_docs_simple(ordered_docs_w_chunks: List[dict], result_count: int) -> dict:
    """Only one highlight is returned
    Args:
        ordered_docs_w_chunks:
    Returns:
    """
    simple_results = []

    for d in ordered_docs_w_chunks:
        if "_source" in d['doc']:
            cleaned = _clean_doc(d['doc']["_source"], doc_id=d['_id'])
        else:
            cleaned = _clean_doc(dict(), doc_id=d['_id'])

        cleaned["_highlights"] = {
            d["chunks"][0]["_source"][TensorField.field_name]: d["chunks"][0]["_source"][
                TensorField.field_content]
        }
        cleaned["_score"] = d["chunks"][0]["_score"]
        simple_results.append(cleaned)
    return {"hits": simple_results[:result_count]}

def boost_score(docs: dict, boosters: dict, searchable_attributes) -> dict:
    """ re-weighs the scores of individual fields
        Args:
            docs:
            boosters: {'field_to_be_boosted': (int, int)}
        """
    to_be_boosted = docs.copy()
    boosted_fields = set()
    if searchable_attributes and boosters:
        if not set(boosters).issubset(set(searchable_attributes)):
            raise errors.InvalidArgError(
        "Boost fieldnames must be a subset of searchable attributes. "
        f"\nSearchable attributes: {searchable_attributes}"
        f"\nBoost: {boosters}"
        )

    for doc_id in list(to_be_boosted.keys()):
        for chunk in to_be_boosted[doc_id]["chunks"]:
            field_name = chunk['_source']['__field_name']
            if field_name in boosters.keys():
                booster = boosters[field_name]
                if len(booster) == 2:
                    # weight and bias are given
                    chunk['_score'] = chunk['_score'] * booster[0] + booster[1]
                else:
                    # only weight is given
                    chunk['_score'] = chunk['_score'] * booster[0]
                boosted_fields.add(field_name)
    return to_be_boosted


def sort_chunks(docs: dict) -> List:
    to_be_sorted = docs.copy()
    for doc_id in list(to_be_sorted.keys()):
        to_be_sorted[doc_id]["chunks"] = sorted(
            to_be_sorted[doc_id]["chunks"], key=lambda x: x["_score"], reverse=True)

    as_list = list(docs.values())
    return sorted(as_list, key=lambda x: x["chunks"][0]["_score"], reverse=True)


def check_index_health(config: Config, index_name: str) -> dict:
    """Checks the health of an index
    Args:
        config: Config
        index_name: str
    Returns:
        dict
    """
    return generate_heath_check_response(config, index_name=index_name)


def check_health(config: Config) -> dict:
    """Check the health of the Marqo-os backend.
    Deprecated in Marqo 1.0.0 and will be removed in future versions.
    Please use check_index_health instead.
    """
    return generate_heath_check_response(config)

def delete_index(config: Config, index_name):
    res = HttpRequests(config).delete(path=index_name)
    if index_name in get_cache():
        del get_cache()[index_name]
    return res


def get_indexes(config: Config):
    res = backend.get_cluster_indices(config=config)

    body = {
        'results': [
            {'index_name': ix} for ix in res
        ]
    }
    return body


def _select_model_from_media_type(media_type: Union[MediaType, str]) -> Union[MlModel, str]:
    if media_type == MediaType.text:
        return MlModel.bert
    elif media_type == MediaType.image:
        return MlModel.clip
    else:
        raise ValueError("_select_model_from_media_type(): "
                         "Received unknown media type: {}".format(media_type))


def get_loaded_models() -> dict:
    available_models = s2_inference.get_available_models()
    message = {"models": []}

    for ix in available_models:
        if isinstance(ix, str):
            message["models"].append({"model_name": ix.split("||")[0], "model_device": ix.split("||")[-1]})
    return message


def eject_model(model_name: str, device: str) -> dict:
    try:
        result = s2_inference.eject_model(model_name, device)
    except s2_inference_errors.ModelNotInCacheError as e:
        raise errors.ModelNotInCacheError(message=str(e))
    return result


def get_cpu_info() -> dict:
    return {
        "cpu_usage_percent": f"{psutil.cpu_percent(1)} %",  # The number 1 is a time interval for CPU usage calculation.
        "memory_used_percent": f"{psutil.virtual_memory()[2]} %",
        # The number 2 is just a index number to get the expected results
        "memory_used_gb": f"{round(psutil.virtual_memory()[3] / 1000000000, 1)}",
        # The number 3 is just a index number to get the expected results
    }


def get_cuda_info() -> dict:
    if torch.cuda.is_available():
        return {"cuda_devices": [{"device_id": _device_id, "device_name": torch.cuda.get_device_name(_device_id),
                                  "memory_used": f"{round(torch.cuda.memory_allocated(_device_id) / 1024 ** 3, 1)} GiB",
                                  "total_memory": f"{round(torch.cuda.get_device_properties(_device_id).total_memory / 1024 ** 3, 1)} GiB"}
                                 for _device_id in range(torch.cuda.device_count())]}

    else:
        raise errors.HardwareCompatabilityError(message=str(
            "ERROR: cuda is not supported in your machine!!"
        ))


def vectorise_multimodal_combination_field(
        field: str, multimodal_object: Dict[str, dict], doc: dict, doc_index: int,
        doc_id:str, device:str, index_info, image_repo, field_map:dict,
        model_auth: Optional[ModelAuth] = None
):
    '''
    This function is used to vectorise multimodal combination field. The field content should
    have the following structure:
    field_conent = {"tensor_field_one" : {"weight":0.5, "parameter": "test-paramater-1"},
                    "tensor_field_two" : {"weight": 0.5, parameter": "test-parameter-2"}},
    Over all this is a simplified version of the vectorise pipeline in add_documents. Specifically,
    1. we don't do any chunking here.
    2. we don't use image repo for concurrent downloading.
    Args:
        field_content: the field content that is a dictionary
        copied: the copied document
        unsuccessful_docs: a list to store all the unsuccessful documents
        total_vectorise_time: total vectorise time in the main body
        doc_index: the index of the document. This is an interator variable `i` in the main body to iterator throught the docs
        doc_id: the document id
        device: device from main body
        index_info: index_info from main body,
        model_auth: Model download authorisation information (if required)
    Returns:
        combo_chunk: the combo_chunk to be appended to the main body
        combo_document_is_valid:  if the document is a valid
        unsuccessful_docs: appended unsucessful_docs
        combo_total_vectorise_time: the vectorise time spent in combo field
        new_fields_from_multimodal_combination: the new fields from multimodal combination field that will be added to
            index properties

    '''
    # field_content = {"tensor_field_one" : {"weight":0.5, "parameter": "test-paramater-1"},
    #                 "tensor_field_two" : {"weight": 0.5, parameter": "test-parameter-2"}},
    combo_document_is_valid = True
    combo_vectorise_time_to_add = 0
    combo_chunk = {}
    unsuccessful_doc_to_append = tuple()
    new_fields_from_multimodal_combination = set()

    # Copy the important mutable objects from main body for safety purpose
    multimodal_object_copy = copy.deepcopy(multimodal_object)

    # 4 lists to store the field name and field content to vectorise.
    text_field_names = []
    text_content_to_vectorise = []

    image_field_names = []
    image_content_to_vectorise = []

    normalize_embeddings = index_info.index_settings[NsField.index_defaults][
        NsField.normalize_embeddings]
    infer_if_image = index_info.index_settings[NsField.index_defaults][
        NsField.treat_urls_and_pointers_as_images]

    if infer_if_image is False:
        text_field_names = list(multimodal_object.keys())
        text_content_to_vectorise = list(multimodal_object.values())
        new_fields_from_multimodal_combination =set([(sub_field_name, _infer_opensearch_data_type(sub_content)) for sub_field_name
        , sub_content in multimodal_object.items()])
    else:
        for sub_field_name, sub_content in multimodal_object.items():
            if isinstance(sub_content, str) and not _is_image(sub_content):
                text_field_names.append(sub_field_name)
                text_content_to_vectorise.append(sub_content)
            else:
                try:
                    if isinstance(sub_content, str) and index_info.index_settings[NsField.index_defaults][
                            NsField.treat_urls_and_pointers_as_images]:
                        if not isinstance(image_repo[sub_content], Exception):
                            image_data = image_repo[sub_content]
                        else:
                            raise s2_inference_errors.S2InferenceError(
                                f"Could not find image found at `{sub_content}`. \n"
                                f"Reason: {str(image_repo[sub_content])}"
                            )
                    else:
                        image_data = sub_content

                    image_content_to_vectorise.append(image_data)
                    image_field_names.append(sub_field_name)

                except s2_inference_errors.S2InferenceError as e:
                    combo_document_is_valid = False
                    unsuccessful_doc_to_append = \
                        (doc_index, {'_id': doc_id, 'error': e.message,
                                     'status': int(errors.InvalidArgError.status_code),
                                     'code': errors.InvalidArgError.code})

                    return combo_chunk, combo_document_is_valid, unsuccessful_doc_to_append, combo_vectorise_time_to_add, new_fields_from_multimodal_combination
            new_fields_from_multimodal_combination.add((sub_field_name, _infer_opensearch_data_type(sub_content)))

    try:
        start_time = timer()
        text_vectors = []
        if len(text_content_to_vectorise) > 0:
            with RequestMetricsStore.for_request().time(f"create_vectors"):
                text_vectors = s2_inference.vectorise(
                    model_name=index_info.model_name,
                    model_properties=index_info.get_model_properties(), content=text_content_to_vectorise,
                    device=device, normalize_embeddings=normalize_embeddings,
                    infer=infer_if_image, model_auth=model_auth
                )
        image_vectors = []
        if len(image_content_to_vectorise) > 0:
            with RequestMetricsStore.for_request().time(f"create_vectors"):
                image_vectors = s2_inference.vectorise(
                    model_name=index_info.model_name,
                    model_properties=index_info.get_model_properties(), content=image_content_to_vectorise,
                    device=device, normalize_embeddings=normalize_embeddings,
                    infer=infer_if_image, model_auth=model_auth
                )
        end_time = timer()
        combo_vectorise_time_to_add += (end_time - start_time)
    except (s2_inference_errors.UnknownModelError,
            s2_inference_errors.InvalidModelPropertiesError,
            s2_inference_errors.ModelLoadError) as model_error:
        raise errors.BadRequestError(
            message=f'Problem vectorising query. Reason: {str(model_error)}',
            link="https://marqo.pages.dev/1.4.0/Models-Reference/dense_retrieval/"
        )
    except s2_inference_errors.S2InferenceError:
        combo_document_is_valid = False
        image_err = errors.InvalidArgError(message=f'Could not process given image: {multimodal_object_copy}')
        unsuccessful_doc_to_append = \
            (doc_index, {'_id': doc_id, 'error': image_err.message, 'status': int(image_err.status_code),
                 'code': image_err.code})

        return combo_chunk, combo_document_is_valid, unsuccessful_doc_to_append, combo_vectorise_time_to_add, new_fields_from_multimodal_combination

    sub_field_name_list = text_field_names + image_field_names
    vectors_list = text_vectors + image_vectors

    if not len(sub_field_name_list) == len(vectors_list):
        raise errors.BatchInferenceSizeError(message=f"Batch inference size does not match content for multimodal field {field}")

    vector_chunk = np.squeeze(np.mean([np.array(vector) * field_map["weights"][sub_field_name] for sub_field_name, vector in zip(sub_field_name_list, vectors_list)], axis=0))

    if normalize_embeddings is True:
        vector_chunk = vector_chunk / np.linalg.norm(vector_chunk)

    vector_chunk = vector_chunk.tolist()

    combo_chunk = dict({
        TensorField.marqo_knn_field: vector_chunk,
        TensorField.field_content: json.dumps(multimodal_object),
        TensorField.field_name: field,
    })
    return combo_chunk, combo_document_is_valid, unsuccessful_doc_to_append, combo_vectorise_time_to_add, new_fields_from_multimodal_combination

def _generate_vector_text_search_query_for_verbose_one(original_body:List[Dict[str, Any]]) -> None:
    """Generate a simplified version of the query body for verbose=1 mode. """
    readable_body = copy.deepcopy(original_body)
    for i, q in enumerate(readable_body):
        if "index" in q:
            continue
        if "query" in q and "nested" in q.get("query"):
            # A normal vector search
            for vec in list(q["query"]["nested"]["query"]["knn"].keys()):
                if "vector" in q["query"]["nested"]["query"]["knn"][vec]:
                    readable_body[i]["query"]["nested"]["query"]["knn"][vec]["vector"] = \
                        readable_body[i]["query"]["nested"]["query"]["knn"][vec]["vector"][:5]
        elif "query" in q and "function_score" in q.get("query"):
            # A score modifier search
            for vec in list(
                    q["query"]["function_score"]["query"]["nested"]["query"]["function_score"]["query"]["knn"].keys()):
                if "vector" in \
                        q["query"]["function_score"]["query"]["nested"]["query"]["function_score"]["query"]["knn"][vec]:
                    readable_body[i]["query"]["function_score"]["query"]["nested"]["query"]["function_score"]["query"][
                        "knn"][vec]["vector"] = \
                        readable_body[i]["query"]["function_score"]["query"]["nested"]["query"]["function_score"][
                            "query"]["knn"][vec]["vector"][:5]
        elif "query" in q and "match_none" in q.get("query"):
            # A dummy search to replace a zero-vector
            pass
        else:
            raise errors.InternalError(f"Marqo encountered an unexpected query format "
                                       f"in `_vector_text_search` when setting `verbose=1`. "
                                       f"The unexpected query is `{q}`")
    return readable_body


def _vector_text_search_query_verbose(verbose: int, body: List[Dict[str, Any]]) -> None:
    """Handle the verbose flag for _vector_text_search queries"""
    print("vector search body:")
    if verbose == 1:
        pprint.pprint(_generate_vector_text_search_query_for_verbose_one(body))
    elif verbose == 2:
        pprint.pprint(body, compact=True)
    else:
        raise errors.InternalError(f"Marqo encountered an unexpected verbose flag = `{verbose}`")


def _create_normal_tensor_search_query(result_count, offset, vector_field, vectorised_text) -> dict:
    search_query = {
        "size": result_count,
        "from": offset,
        "query": {
            "nested": {
                "path": TensorField.chunks,
                "inner_hits": {
                    "_source": {
                        "include": ["__chunks.__field_content", "__chunks.__field_name"]
                    }
                },
                "query": {
                    "knn": {
                        f"{TensorField.chunks}.{vector_field}": {
                            "vector": vectorised_text,
                            "k": result_count + offset
                        }
                    }
                },
                "score_mode": "max"
            }
        },
        "_source": {
            "exclude": ["__chunks.__vector_*"]
        }
    }
    return search_query


def _create_score_modifiers_tensor_search_query(score_modifiers, result_count, offset, vector_field, vectorised_text) -> dict:
    script_score = score_modifiers.to_script_score()
    search_query = {
        "size": result_count,
        "from": offset,
        "query": {
            "function_score": {
                "query": {
                    "nested": {
                        "path": TensorField.chunks,
                        "inner_hits": {
                            "_source": {
                                "include": ["__chunks.__field_content", "__chunks.__field_name"]
                            }
                        },
                        "query": {
                            "function_score": {
                                "query": {
                                    "knn": {
                                        f"{TensorField.chunks}.{vector_field}": {
                                            "vector": vectorised_text,
                                            "k": result_count + offset
                                        }
                                    }
                                },
                                "functions": [
                                    {
                                        "script_score": {
                                            "script": {
                                                **script_score
                                            }
                                        }
                                    }
                                ],
                                "boost_mode": "replace"
                            }
                        },
                        "score_mode": "max"
                    }
                },
            }
        },
        "_source": {
            "exclude": ["__chunks.__vector_*"]
        }
    }
    return search_query


def _create_dummy_query_for_zero_vector_search() -> dict:
    """A dummy search query that returns no results. Used when the query vector is all zeros."""
    search_query = {
        "query": {
            "match_none": {}
        }
    }
    return search_query


def delete_documents(config: Config, index_name: str, doc_ids: List[str], auto_refresh):
    """Delete documents from the Marqo index with the given doc_ids """
    return delete_docs.delete_documents(
        config=config,
        del_request=MqDeleteDocsRequest(
            index_name=index_name,
            document_ids=doc_ids,
            auto_refresh=auto_refresh)
    )
