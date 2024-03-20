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
import typing
import uuid
from collections import defaultdict
from contextlib import ExitStack
from timeit import default_timer as timer
from typing import List, Optional, Union, Iterable, Sequence, Dict, Any, Tuple

import numpy as np
import psutil
import torch.cuda
from PIL import Image

import marqo.core.unstructured_vespa_index.common as unstructured_common
from marqo.api import exceptions as api_exceptions
from marqo.api import exceptions as errors
# We depend on _httprequests.py for now, but this may be replaced in the future, as
# _httprequests.py is designed for the client
from marqo.config import Config
from marqo.core import constants
from marqo.core import exceptions as core_exceptions
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import IndexType
from marqo.core.models.marqo_index import MarqoIndex, FieldType, UnstructuredMarqoIndex, StructuredMarqoIndex
from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoLexicalQuery
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
from marqo.core.unstructured_vespa_index import unstructured_validation as unstructured_index_add_doc_validation
from marqo.core.unstructured_vespa_index.unstructured_vespa_index import UnstructuredVespaIndex
from marqo.core.vespa_index import for_marqo_index as vespa_index_factory
from marqo.s2_inference import errors as s2_inference_errors
from marqo.s2_inference import s2_inference
from marqo.s2_inference.clip_utils import _is_image
from marqo.s2_inference.processing import image as image_processor
from marqo.s2_inference.processing import text as text_processor
from marqo.s2_inference.reranking import rerank
from marqo.tensor_search import delete_docs
from marqo.tensor_search import enums
from marqo.tensor_search import index_meta_cache
from marqo.tensor_search import utils, validation, add_docs
from marqo.tensor_search.enums import (
    Device, TensorField, SearchMethod, EnvVars,
    MappingsObjectType
)
from marqo.tensor_search.index_meta_cache import get_cache, get_index
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.models.api_models import BulkSearchQueryEntity, ScoreModifier
from marqo.tensor_search.models.delete_docs_objects import MqDeleteDocsRequest
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.search import Qidx, JHash, SearchContext, VectorisedJobs, VectorisedJobPointer
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.tensor_search.tensor_search_logging import get_logger
from marqo.vespa.exceptions import VespaStatusError
from marqo.vespa.models import VespaDocument, FeedBatchResponse, QueryResult
from marqo import marqo_docs

logger = get_logger(__name__)


def add_documents(config: Config, add_docs_params: AddDocsParams):
    """
    Args:
        config: Config object
        add_docs_params: add_documents()'s parameters
    """
    try:
        marqo_index = index_meta_cache.get_index(
            config=config, index_name=add_docs_params.index_name, force_refresh=True
        )

    # TODO: raise core_exceptions.IndexNotFoundError instead (fix associated tests)
    except api_exceptions.IndexNotFoundError:
        raise api_exceptions.IndexNotFoundError(
            f"Cannot add documents to non-existent index {add_docs_params.index_name}")

    if isinstance(marqo_index, UnstructuredMarqoIndex):
        return _add_documents_unstructured(config, add_docs_params, marqo_index)
    elif isinstance(marqo_index, StructuredMarqoIndex):
        return _add_documents_structured(config, add_docs_params, marqo_index)
    else:
        raise api_exceptions.InternalError(f"Unknown index type {type(marqo_index)}")


def _add_documents_unstructured(config: Config, add_docs_params: AddDocsParams, marqo_index: UnstructuredMarqoIndex):
    # ADD DOCS TIMER-LOGGER (3)
    vespa_client = config.vespa_client
    unstructured_vespa_index = UnstructuredVespaIndex(marqo_index)
    index_model_dimensions = marqo_index.model.get_dimension()

    RequestMetricsStore.for_request().start("add_documents.processing_before_vespa")

    unstructured_index_add_doc_validation.validate_tensor_fields(add_docs_params.tensor_fields)

    multimodal_sub_fields = []
    if add_docs_params.mappings is not None:
        unstructured_index_add_doc_validation.validate_mappings_object_format(add_docs_params.mappings)
        for field_name, mapping in add_docs_params.mappings.items():
            if mapping.get("type", None) == enums.MappingsObjectType.multimodal_combination:
                multimodal_sub_fields.extend(mapping["weights"].keys())

    t0 = timer()
    bulk_parent_dicts = []

    if len(add_docs_params.docs) == 0:
        raise errors.BadRequestError(message="Received empty add documents request")

    unsuccessful_docs = []
    total_vectorise_time = 0
    batch_size = len(add_docs_params.docs)
    image_repo = {}

    docs, doc_ids = config.document.remove_duplicated_documents(add_docs_params.docs)

    with ExitStack() as exit_stack:
        if marqo_index.treat_urls_and_pointers_as_images:
            with RequestMetricsStore.for_request().time(
                    "image_download.full_time",
                    lambda t: logger.debug(
                        f"add_documents image download: took {t:.3f}ms to concurrently download "
                        f"images for {batch_size} docs using {add_docs_params.image_download_thread_count} threads"
                    )
            ):
                # TODO - Refactor this part to make it more readable
                # We need to pass the subfields to the image downloader, so that it can download the images in the
                # multimodal subfields even if the subfield is not a tensor_field
                tensor_fields_and_multimodal_subfields = copy.deepcopy(add_docs_params.tensor_fields) \
                    if add_docs_params.tensor_fields else []
                tensor_fields_and_multimodal_subfields.extend(multimodal_sub_fields)
                image_repo = exit_stack.enter_context(
                    add_docs.download_images(docs=docs,
                                             thread_count=add_docs_params.image_download_thread_count,
                                             tensor_fields=tensor_fields_and_multimodal_subfields,
                                             image_download_headers=add_docs_params.image_download_headers)
                )

        if add_docs_params.use_existing_tensors:
            existing_docs_dict: Dict[str, dict] = {}
            if len(doc_ids) > 0:
                existing_docs = _get_marqo_documents_by_ids(config, marqo_index.name, doc_ids, ignore_invalid_ids=True)
                for doc in existing_docs:
                    id = doc["_id"]
                    if id in existing_docs_dict:
                        raise errors.InternalError(f"Received duplicate documents for ID {id} from Vespa")
                    existing_docs_dict[id] = doc

                logger.debug(f"Found {len(existing_docs_dict)} existing docs")

        for i, doc in enumerate(docs):

            copied = copy.deepcopy(doc)

            document_is_valid = True

            doc_id = None
            try:
                validation.validate_doc(doc)

                if add_docs_params.mappings and multimodal_sub_fields:
                    unstructured_index_add_doc_validation.validate_coupling_of_mappings_and_doc(
                        doc, add_docs_params.mappings, multimodal_sub_fields
                    )

                if "_id" in doc:
                    doc_id = validation.validate_id(doc["_id"])
                    del copied["_id"]
                else:
                    doc_id = str(uuid.uuid4())

                [unstructured_index_add_doc_validation.validate_field_name(field) for field in copied]

            except errors.__InvalidRequestError as err:
                unsuccessful_docs.append(
                    (i, {'_id': doc_id if doc_id is not None else '',
                         'error': err.message, 'status': int(err.status_code), 'code': err.code})
                )
                continue

            processed_tensor_fields: List[str] = []
            embeddings_list: List[str] = []

            for field in copied:

                is_tensor_field = utils.is_tensor_field(field, add_docs_params.tensor_fields)

                try:
                    field_content = unstructured_vespa_index.validate_field_content(
                        field_content=copied[field],
                        is_tensor_field=is_tensor_field
                    )
                    # Used to validate custom_vector field or any other new dict field type
                    if isinstance(field_content, dict):
                        field_content = validation.validate_dict(
                            field=field, field_content=field_content,
                            is_non_tensor_field=not is_tensor_field,
                            mappings=add_docs_params.mappings, index_model_dimensions=index_model_dimensions)
                except (errors.InvalidArgError, core_exceptions.MarqoDocumentParsingError) as err:
                    document_is_valid = False
                    unsuccessful_docs.append(
                        (i, {'_id': doc_id, 'error': err.message, 'status': int(err.status_code),
                             'code': err.code})
                    )
                    break

                # Proceed from here only for tensor fields
                if not is_tensor_field:
                    continue

                # chunks generated by processing this field for this doc:
                chunks: List[str] = []
                embeddings: List[List[float]] = []

                # 4 current options for chunking/vectorisation behavior:
                # A) field type is custom_vector -> no chunking or vectorisation
                # B) use_existing_tensors=True and field content hasn't changed -> no chunking or vectorisation
                # C) field type is standard -> chunking and vectorisation
                # D) field type is multimodal -> use vectorise_multimodal_combination_field (does chunking and vectorisation)
                # Do step D regardless. It will generate separate chunks for multimodal.

                # A) Calculate custom vector field logic here. It should ignore use_existing_tensors, as this step has no vectorisation.
                document_dict_field_type = add_docs.determine_document_dict_field_type(field, field_content,
                                                                                       add_docs_params.mappings)
                if document_dict_field_type == FieldType.CustomVector:
                    # Generate exactly 1 chunk with the custom vector.
                    chunks = [f"{field}::{copied[field]['content']}"]
                    embeddings = [copied[field]["vector"]]

                    # Update parent document (copied) to fit new format. Use content (text) to replace input dict
                    copied[field] = field_content["content"]
                    logger.debug(f"Custom vector field {field} added as 1 chunk.")

                # B) Use existing tensors if available and existing content did not change.
                elif (
                        add_docs_params.use_existing_tensors and
                        doc_id in existing_docs_dict and
                        field in existing_docs_dict[doc_id] and
                        existing_docs_dict[doc_id][field] == field_content
                ):
                    if (
                            constants.MARQO_DOC_TENSORS in existing_docs_dict[doc_id] and
                            field in existing_docs_dict[doc_id][constants.MARQO_DOC_TENSORS]
                    ):
                        chunks: List[str] = [f"{field}::{content}" for content in
                                             existing_docs_dict[doc_id][constants.MARQO_DOC_TENSORS][field][
                                                 constants.MARQO_DOC_CHUNKS]]
                        embeddings: List[List[float]] = [existing_docs_dict[doc_id][constants.MARQO_DOC_TENSORS][field][
                                                             constants.MARQO_DOC_EMBEDDINGS]]
                        logger.debug(f"Using existing tensors for field {field} for doc {doc_id}")
                    else:
                        # Happens if this wasn't a tensor field last time we indexed this doc
                        logger.debug(f"Found document but not tensors for field {field} for doc {doc_id}. "
                                     f"Is this a new tensor field?")

                # C) field type is standard
                if len(chunks) == 0:  # Not using existing tensors or didn't find it
                    if isinstance(field_content, (str, Image.Image)):
                        # 1. check if urls should be downloaded -> "treat_pointers_and_urls_as_images":True
                        # 2. check if it is a url or pointer
                        # 3. If yes in 1 and 2, download blindly (without type)
                        # 4. Determine media type of downloaded
                        # 5. load correct media type into memory -> PIL (images), videos (), audio (torchaudio)
                        # 6. if chunking -> then add the extra chunker

                        if isinstance(field_content, str) and not _is_image(field_content):
                            # text processing pipeline:
                            split_by = marqo_index.text_preprocessing.split_method.value
                            split_length = marqo_index.text_preprocessing.split_length
                            split_overlap = marqo_index.text_preprocessing.split_overlap
                            content_chunks: List[str] = text_processor.split_text(field_content, split_by=split_by,
                                                                                  split_length=split_length,
                                                                                  split_overlap=split_overlap)
                            text_chunks = content_chunks
                        else:
                            # TODO put the logic for getting field parameters into a function and add per field options
                            image_method = marqo_index.image_preprocessing.patch_method

                            # the chunk_image contains the no-op logic as of now - method = None will be a no-op
                            try:
                                # in the future, if we have different chunking methods, make sure we catch possible
                                # errors of different types generated here, too.
                                if isinstance(field_content, str) and marqo_index.treat_urls_and_pointers_as_images:
                                    if not isinstance(image_repo[field_content], Exception):
                                        image_data = image_repo[field_content]
                                    else:
                                        raise s2_inference_errors.S2InferenceError(
                                            f"Could not find image found at `{field_content}`. \n"
                                            f"Reason: {str(image_repo[field_content])}"
                                        )
                                else:
                                    image_data = field_content
                                if image_method is not None:
                                    content_chunks, text_chunks = image_processor.chunk_image(
                                        image_data, device=add_docs_params.device, method=image_method.value)
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

                        normalize_embeddings = marqo_index.normalize_embeddings

                        try:
                            # in the future, if we have different underlying vectorising methods, make sure we catch possible
                            # errors of different types generated here, too.

                            # ADD DOCS TIMER-LOGGER (4)
                            start_time = timer()
                            with RequestMetricsStore.for_request().time(f"add_documents.create_vectors"):
                                vector_chunks = s2_inference.vectorise(
                                    model_name=marqo_index.model.name,
                                    model_properties=marqo_index.model.get_properties(), content=content_chunks,
                                    device=add_docs_params.device, normalize_embeddings=normalize_embeddings,
                                    infer=marqo_index.treat_urls_and_pointers_as_images,
                                    model_auth=add_docs_params.model_auth
                                )

                            end_time = timer()
                            total_vectorise_time += (end_time - start_time)
                        except (s2_inference_errors.UnknownModelError,
                                s2_inference_errors.InvalidModelPropertiesError,
                                s2_inference_errors.ModelLoadError,
                                s2_inference.ModelDownloadError) as model_error:
                            raise errors.BadRequestError(
                                message=f'Problem vectorising query. Reason: {str(model_error)}',
                                link=marqo_docs.list_of_models()
                            )
                        except s2_inference_errors.VectoriseError as e:
                            document_is_valid = False
                            unsuccessful_docs.append(
                                (i, {'_id': doc_id, 'error': e.message,
                                     'status': int(errors.InvalidArgError.status_code),
                                     'code': errors.InvalidArgError.code})
                            )
                            break
                        except s2_inference_errors.S2InferenceError:
                            document_is_valid = False
                            image_err = errors.InvalidArgError(
                                message=f'Could not process given image: {field_content}')
                            unsuccessful_docs.append(
                                (i, {'_id': doc_id, 'error': image_err.message, 'status': int(image_err.status_code),
                                     'code': image_err.code})
                            )
                            break

                        if len(vector_chunks) != len(text_chunks):
                            raise RuntimeError(
                                f"the input content after preprocessing and its vectorized counterparts must be the same length."
                                f"received text_chunks={len(text_chunks)} and vector_chunks={len(vector_chunks)}. "
                                f"check the preprocessing functions and try again. ")

                        chunks: List[str] = [f"{field}::{text_chunk}" for text_chunk in text_chunks]
                        embeddings: List[List[float]] = vector_chunks

                        assert len(chunks) == len(embeddings), "Chunks and embeddings must be the same length"
                    else:
                        raise errors.InvalidArgError(f'Invalid type {type(field_content)} for tensor field {field}')

                processed_tensor_fields.extend(chunks)
                embeddings_list.extend(embeddings)

            # All the plain tensor/non-tensor fields are processed, now we process the multimodal fields
            if document_is_valid and add_docs_params.mappings:
                multimodal_mappings: Dict[str, Dict] = utils.extract_multimodal_mappings(add_docs_params.mappings)

                for field_name, multimodal_params in multimodal_mappings.items():
                    if not utils.is_tensor_field(field_name, add_docs_params.tensor_fields):
                        raise errors.InvalidArgError(f"Multimodal field {field_name} must be a tensor field")

                    field_content: Dict[str, str] = utils.extract_multimodal_content(copied, multimodal_params)

                    combo_chunk: Optional[str] = None

                    if (
                            add_docs_params.use_existing_tensors and
                            doc_id in existing_docs_dict
                    ):
                        existing_doc = existing_docs_dict[doc_id]
                        current_field_contents = utils.extract_multimodal_content(existing_doc, multimodal_params)
                        if (
                                field_content == current_field_contents and
                                unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS in existing_doc and
                                field_name in existing_doc[unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS] and
                                existing_doc[unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS][
                                    field_name] == multimodal_params and
                                field_name in existing_doc[constants.MARQO_DOC_TENSORS]
                        ):
                            combo_chunk = f"{field_name}::{existing_doc[constants.MARQO_DOC_TENSORS][field_name][constants.MARQO_DOC_CHUNKS][0]}"
                            combo_embeddings = existing_doc[constants.MARQO_DOC_TENSORS][field_name][
                                constants.MARQO_DOC_EMBEDDINGS]

                            if unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS not in copied:
                                copied[unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS] = {}
                            copied[unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS][field_name] = json.dumps(
                                multimodal_params)
                            processed_tensor_fields.append(combo_chunk)
                            embeddings_list.append(combo_embeddings)

                            logger.debug(
                                f"Using existing tensors for multimodal combination field {field_name} for doc {doc_id}"
                            )
                        else:
                            logger.debug(
                                f'Not using existing tensors for multimodal combination field {field_name} for '
                                f'doc {doc_id} because field content or config has changed')

                    # Use_existing tensor does not apply, or we didn't find it, then we vectorise
                    if combo_chunk is None:
                        if field_content:  # Check if the subfields are present
                            (combo_chunk, combo_embeddings, combo_document_is_valid,
                             unsuccessful_doc_to_append,
                             combo_vectorise_time_to_add) = vectorise_multimodal_combination_field_unstructured(
                                field_name,
                                field_content, i, doc_id, add_docs_params.device, marqo_index,
                                image_repo, multimodal_params, model_auth=add_docs_params.model_auth)

                            total_vectorise_time = total_vectorise_time + combo_vectorise_time_to_add
                            if combo_document_is_valid is False:
                                document_is_valid = False
                                unsuccessful_docs.append(unsuccessful_doc_to_append)
                                break
                            else:

                                if unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS not in copied:
                                    copied[unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS] = {}

                                copied[unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS][field_name] = json.dumps(
                                    multimodal_params)
                                processed_tensor_fields.append(combo_chunk)
                                embeddings_list.append(combo_embeddings)
                        else:
                            continue

            if document_is_valid:
                if processed_tensor_fields:
                    processed_marqo_embeddings = {k: v for k, v in enumerate(embeddings_list)}
                    assert len(processed_tensor_fields) == len(
                        processed_marqo_embeddings), "Chunks and embeddings must be the same length"
                    copied[constants.MARQO_DOC_CHUNKS] = processed_tensor_fields
                    copied[constants.MARQO_DOC_EMBEDDINGS] = processed_marqo_embeddings
                copied[constants.MARQO_DOC_ID] = doc_id
                bulk_parent_dicts.append(copied)

    total_preproc_time = 0.001 * RequestMetricsStore.for_request().stop(
        "add_documents.processing_before_vespa")
    logger.debug(
        f"      add_documents pre-processing: took {(total_preproc_time):.3f}s total for {batch_size} docs, "
        f"for an average of {(total_preproc_time / batch_size):.3f}s per doc.")

    logger.debug(f"          add_documents vectorise: took {(total_vectorise_time):.3f}s for {batch_size} docs, "
                 f"for an average of {(total_vectorise_time / batch_size):.3f}s per doc.")

    if bulk_parent_dicts:
        vespa_docs = [
            VespaDocument(**unstructured_vespa_index.to_vespa_document(marqo_document=doc))
            for doc in bulk_parent_dicts
        ]
        # ADD DOCS TIMER-LOGGER (5)
        start_time_5 = timer()
        with RequestMetricsStore.for_request().time("add_documents.vespa._bulk"):
            index_responses = vespa_client.feed_batch(vespa_docs, marqo_index.schema_name)

        end_time_5 = timer()
        total_http_time = end_time_5 - start_time_5
        logger.debug(
            f"      add_documents roundtrip: took {(total_http_time):.3f}s to send {batch_size} "
            f"docs (roundtrip) to vector store, "
            f"for an average of {(total_http_time / batch_size):.3f}s per doc.")
    else:
        index_responses = None

    with RequestMetricsStore.for_request().time("add_documents.postprocess"):
        t1 = timer()

        def translate_add_doc_response(responses: Optional[FeedBatchResponse], time_diff: float) -> dict:
            """translates Vespa response dict into Marqo dict"""
            result_dict = {}
            new_items = []

            if responses is not None:
                result_dict['errors'] = responses.errors

                for resp in responses.responses:
                    id = resp.id.split('::')[-1] if resp.id else None
                    new_items.append({'status': resp.status})
                    if id:
                        new_items[-1].update({'_id': id})
                    if resp.message:
                        new_items[-1].update({'message': resp.message})

            if unsuccessful_docs:
                result_dict['errors'] = True

            for loc, error_info in unsuccessful_docs:
                new_items.insert(loc, error_info)

            result_dict["processingTimeMs"] = time_diff * 1000
            result_dict["index_name"] = add_docs_params.index_name
            result_dict["items"] = new_items

            return result_dict

        return translate_add_doc_response(index_responses, time_diff=t1 - t0)


def _add_documents_structured(config: Config, add_docs_params: AddDocsParams, marqo_index: StructuredMarqoIndex):
    # ADD DOCS TIMER-LOGGER (3)
    vespa_client = config.vespa_client
    vespa_index = StructuredVespaIndex(marqo_index)
    index_model_dimensions = marqo_index.model.get_dimension()

    RequestMetricsStore.for_request().start("add_documents.processing_before_vespa")

    if add_docs_params.tensor_fields is not None:
        raise api_exceptions.InvalidArgError('Cannot specify `tensorFields` for a structured index')

    if add_docs_params.mappings is not None:
        validation.validate_mappings_object(
            add_docs_params.mappings,
            marqo_index
        )
    t0 = timer()
    bulk_parent_dicts: List[VespaDocument] = []

    if len(add_docs_params.docs) == 0:
        raise api_exceptions.BadRequestError(message="Received empty add documents request")

    unsuccessful_docs = []
    total_vectorise_time = 0
    batch_size = len(add_docs_params.docs)  # use length before deduplication
    image_repo = {}

    # Deduplicate docs, keep the latest
    docs, doc_ids = config.document.remove_duplicated_documents(add_docs_params.docs)

    with ExitStack() as exit_stack:
        image_fields = [field.name for field in marqo_index.field_map_by_type[FieldType.ImagePointer]]

        if image_fields:
            with RequestMetricsStore.for_request().time(
                    "image_download.full_time",
                    lambda t: logger.debug(
                        f"add_documents image download: took {t:.3f}ms to concurrently download "
                        f"images for {batch_size} docs using {add_docs_params.image_download_thread_count} threads"
                    )
            ):

                if '_id' in image_fields:
                    raise api_exceptions.BadRequestError(message="`_id` field cannot be an image pointer field.")

                image_repo = exit_stack.enter_context(
                    add_docs.download_images(docs=docs, thread_count=add_docs_params.image_download_thread_count,
                                             tensor_fields=image_fields,
                                             image_download_headers=add_docs_params.image_download_headers)
                )

        if add_docs_params.use_existing_tensors:
            existing_docs_dict: Dict[str, dict] = {}
            if len(doc_ids) > 0:
                existing_docs = _get_marqo_documents_by_ids(config, marqo_index.name, doc_ids, ignore_invalid_ids=True)
                for doc in existing_docs:
                    if not isinstance(doc, dict):
                        continue

                    id = doc["_id"]
                    if id in existing_docs_dict:
                        raise api_exceptions.InternalError(f"Received duplicate documents for ID {id} from Vespa")
                    existing_docs_dict[id] = doc

                logger.debug(f"Found {len(existing_docs_dict)} existing docs")

        for i, doc in enumerate(docs):
            copied = copy.deepcopy(doc)

            document_is_valid = True

            doc_id = None
            try:
                validation.validate_doc(doc)

                if "_id" in doc:
                    doc_id = validation.validate_id(doc["_id"])
                    del copied["_id"]
                else:
                    doc_id = str(uuid.uuid4())

                [validation.validate_field_name(field) for field in copied]
            except api_exceptions.__InvalidRequestError as err:
                unsuccessful_docs.append(
                    (i, {'_id': doc_id if doc_id is not None else '',
                         'error': err.message, 'status': int(err.status_code), 'code': err.code})
                )
                continue

            processed_tensor_fields = {}
            for field in copied:
                marqo_field = marqo_index.field_map.get(field)
                tensor_field = marqo_index.tensor_field_map.get(field)
                is_tensor_field = tensor_field is not None
                if not marqo_field:
                    message = (f"Field {field} is not a valid field for structured index {add_docs_params.index_name}. "
                               f"Valid fields are: {', '.join(marqo_index.field_map.keys())}")
                    document_is_valid = False
                    unsuccessful_docs.append(
                        (i, {'_id': doc_id, 'error': message, 'status': int(api_exceptions.InvalidArgError.status_code),
                             'code': api_exceptions.InvalidArgError.code})
                    )
                    break
                if marqo_field.type == FieldType.MultimodalCombination:
                    message = f"Field {field} is a multimodal combination field and cannot be assigned a value."
                    document_is_valid = False
                    unsuccessful_docs.append(
                        (i, {'_id': doc_id, 'error': message, 'status': int(api_exceptions.InvalidArgError.status_code),
                             'code': api_exceptions.InvalidArgError.code})
                    )
                    break

                try:
                    field_content = validation.validate_field_content(
                        field_content=copied[field],
                        is_non_tensor_field=not is_tensor_field
                    )
                    # Used to validate custom_vector field or any other new dict field type
                    if isinstance(field_content, dict):
                        field_content = validation.validate_dict(
                            field=field, field_content=field_content,
                            is_non_tensor_field=not is_tensor_field,
                            mappings=add_docs_params.mappings, index_model_dimensions=index_model_dimensions,
                            structured_field_type=marqo_field.type)
                except api_exceptions.InvalidArgError as err:
                    document_is_valid = False
                    unsuccessful_docs.append(
                        (i, {'_id': doc_id, 'error': err.message, 'status': int(err.status_code),
                             'code': err.code})
                    )
                    break

                # Proceed from here only for tensor fields
                if not tensor_field:
                    continue

                # chunks generated by processing this field for this doc:
                chunks = []
                embeddings = []

                # 4 current options for chunking/vectorisation behavior:
                # A) field type is custom_vector -> no chunking or vectorisation
                # B) use_existing_tensors=True and field content hasn't changed -> no chunking or vectorisation
                # C) field type is standard -> chunking and vectorisation
                # D) field type is multimodal -> use vectorise_multimodal_combination_field (does chunking and vectorisation)

                # A) Calculate custom vector field logic here. It should ignore use_existing_tensors, as this step has no vectorisation.
                if marqo_field.type == FieldType.CustomVector:
                    # Generate exactly 1 chunk with the custom vector.
                    chunks = [copied[field]['content']]
                    embeddings = [copied[field]["vector"]]

                    # Update parent document (copied) to fit new format. Use content (text) to replace input dict
                    copied[field] = field_content["content"]
                    logger.debug(f"Custom vector field {field} added as 1 chunk.")

                # B) Use existing tensors if available and existing content did not change.
                elif (
                        add_docs_params.use_existing_tensors and
                        doc_id in existing_docs_dict and
                        field in existing_docs_dict[doc_id] and
                        existing_docs_dict[doc_id][field] == field_content
                ):
                    if (
                            constants.MARQO_DOC_TENSORS in existing_docs_dict[doc_id] and
                            field in existing_docs_dict[doc_id][constants.MARQO_DOC_TENSORS]
                    ):
                        chunks = existing_docs_dict[doc_id][constants.MARQO_DOC_TENSORS][field][
                            constants.MARQO_DOC_CHUNKS]
                        embeddings = existing_docs_dict[doc_id][constants.MARQO_DOC_TENSORS][field][
                            constants.MARQO_DOC_EMBEDDINGS]
                        logger.debug(f"Using existing tensors for field {field} for doc {doc_id}")
                    else:
                        # Happens if this wasn't a tensor field last time we indexed this doc
                        logger.debug(f"Found document but not tensors for field {field} for doc {doc_id}. "
                                     f"Is this a new tensor field?")

                if len(chunks) == 0:  # Not using existing tensors or didn't find it
                    if isinstance(field_content, (str, Image.Image)):

                        # TODO: better/consistent handling of a no-op for processing (but still vectorize)

                        # 1. check if urls should be downloaded -> "treat_pointers_and_urls_as_images":True
                        # 2. check if it is a url or pointer
                        # 3. If yes in 1 and 2, download blindly (without type)
                        # 4. Determine media type of downloaded
                        # 5. load correct media type into memory -> PIL (images), videos (), audio (torchaudio)
                        # 6. if chunking -> then add the extra chunker

                        if isinstance(field_content, str) and not _is_image(field_content):
                            # text processing pipeline:
                            split_by = marqo_index.text_preprocessing.split_method.value
                            split_length = marqo_index.text_preprocessing.split_length
                            split_overlap = marqo_index.text_preprocessing.split_overlap
                            content_chunks = text_processor.split_text(field_content, split_by=split_by,
                                                                       split_length=split_length,
                                                                       split_overlap=split_overlap)
                            text_chunks = content_chunks
                        else:
                            # TODO put the logic for getting field parameters into a function and add per field options
                            image_method = marqo_index.image_preprocessing.patch_method

                            # the chunk_image contains the no-op logic as of now - method = None will be a no-op
                            try:
                                # in the future, if we have different chunking methods, make sure we catch possible
                                # errors of different types generated here, too.
                                if isinstance(field_content, str) and field in image_fields:
                                    if not isinstance(image_repo[field_content], Exception):
                                        image_data = image_repo[field_content]
                                    else:
                                        raise s2_inference_errors.S2InferenceError(
                                            f"Could not find image found at `{field_content}`. \n"
                                            f"Reason: {str(image_repo[field_content])}"
                                        )
                                else:
                                    image_data = field_content
                                if image_method is not None:
                                    content_chunks, text_chunks = image_processor.chunk_image(
                                        image_data, device=add_docs_params.device, method=image_method.value)
                                else:
                                    # if we are not chunking, then we set the chunks as 1-len lists
                                    # content_chunk is the PIL image
                                    # text_chunk refers to URL
                                    content_chunks, text_chunks = [image_data], [field_content]
                            except s2_inference_errors.S2InferenceError as e:
                                document_is_valid = False
                                unsuccessful_docs.append(
                                    (i, {'_id': doc_id, 'error': e.message,
                                         'status': int(api_exceptions.InvalidArgError.status_code),
                                         'code': api_exceptions.InvalidArgError.code})
                                )
                                break

                        normalize_embeddings = marqo_index.normalize_embeddings

                        try:
                            # in the future, if we have different underlying vectorising methods, make sure we catch possible
                            # errors of different types generated here, too.

                            # ADD DOCS TIMER-LOGGER (4)
                            start_time = timer()
                            with RequestMetricsStore.for_request().time(f"add_documents.create_vectors"):
                                vector_chunks = s2_inference.vectorise(
                                    model_name=marqo_index.model.name,
                                    model_properties=marqo_index.model.get_properties(), content=content_chunks,
                                    device=add_docs_params.device, normalize_embeddings=normalize_embeddings,
                                    infer=marqo_field.type == FieldType.ImagePointer,
                                    model_auth=add_docs_params.model_auth
                                )

                            end_time = timer()
                            total_vectorise_time += (end_time - start_time)
                        except (s2_inference_errors.UnknownModelError,
                                s2_inference_errors.InvalidModelPropertiesError,
                                s2_inference_errors.ModelLoadError,
                                s2_inference.ModelDownloadError) as model_error:
                            raise api_exceptions.BadRequestError(
                                message=f'Problem vectorising query. Reason: {str(model_error)}',
                                link=marqo_docs.list_of_models()
                            )
                        except s2_inference_errors.S2InferenceError as e:
                            document_is_valid = False
                            unsuccessful_docs.append(
                                (i, {'_id': doc_id, 'error': e.message, 'status': errors.InvalidArgError.status_code,
                                     'code': errors.InvalidArgError.code})
                            )
                            break

                        if len(vector_chunks) != len(text_chunks):
                            raise RuntimeError(
                                f"the input content after preprocessing and its vectorized counterparts must be the same length."
                                f"received text_chunks={len(text_chunks)} and vector_chunks={len(vector_chunks)}. "
                                f"check the preprocessing functions and try again. ")

                        chunks = text_chunks
                        embeddings = vector_chunks

                    else:
                        document_is_valid = False
                        e = api_exceptions.InvalidArgError(
                            f'Invalid type {type(field_content)} for tensor field {field}')
                        unsuccessful_docs.append(
                            (i, {'_id': doc_id, 'error': e.message,
                                 'status': int(api_exceptions.InvalidArgError.status_code),
                                 'code': api_exceptions.InvalidArgError.code})
                        )
                        break

                # Add chunks_to_append along with doc metadata to total chunks
                processed_tensor_fields[tensor_field.name] = {}
                processed_tensor_fields[tensor_field.name][constants.MARQO_DOC_CHUNKS] = chunks
                processed_tensor_fields[tensor_field.name][constants.MARQO_DOC_EMBEDDINGS] = embeddings

            # Multimodal fields haven't been processed yet, so we do that here
            if document_is_valid:  # No need to process multimodal fields if the document is invalid
                for tensor_field in marqo_index.tensor_fields:
                    marqo_field = marqo_index.field_map[tensor_field.name]
                    if marqo_field.type == FieldType.MultimodalCombination:
                        field_name = tensor_field.name
                        field_content = {
                            dependent_field: copied[dependent_field]
                            for dependent_field in marqo_field.dependent_fields if dependent_field in copied
                        }
                        if not field_content:
                            # None of the fields are present in the document, so we skip this multimodal field
                            continue

                        if (
                                add_docs_params.mappings is not None and
                                field_name in add_docs_params.mappings and
                                add_docs_params.mappings[field_name]["type"] == FieldType.MultimodalCombination
                        ):
                            mappings = add_docs_params.mappings[field_name]
                            # Record custom weights in the document
                            copied[field_name] = mappings['weights']
                            logger.debug(f'Using custom weights for multimodal combination field {field_name}')
                        else:
                            mappings = {
                                'weights': marqo_field.dependent_fields
                            }
                            logger.debug(f'Using default weights for multimodal combination field {field_name}: '
                                         f'{marqo_field.dependent_fields}')

                        chunks = []
                        embeddings = []

                        if (
                                add_docs_params.use_existing_tensors and
                                doc_id in existing_docs_dict
                        ):
                            existing_doc = existing_docs_dict[doc_id]
                            current_field_contents = {
                                dependent_field: existing_doc.get(dependent_field)
                                for dependent_field in marqo_field.dependent_fields if dependent_field in copied
                            }
                            current_weights = existing_doc.get(field_name) or marqo_field.dependent_fields
                            if (
                                    field_content == current_field_contents and
                                    current_weights == mappings['weights'] and
                                    field_name in existing_doc[constants.MARQO_DOC_TENSORS]
                            ):
                                chunks = existing_doc[constants.MARQO_DOC_TENSORS][field_name][
                                    constants.MARQO_DOC_CHUNKS]
                                embeddings = existing_doc[constants.MARQO_DOC_TENSORS][field_name][
                                    constants.MARQO_DOC_EMBEDDINGS]
                                logger.debug(
                                    f"Using existing tensors for multimodal combination field {field_name} for doc {doc_id}"
                                )
                            else:
                                logger.debug(
                                    f'Not using existing tensors for multimodal combination field {field_name} for '
                                    f'doc {doc_id} because field content or config has changed')

                        if len(chunks) == 0:  # Not using existing tensors or didn't find it
                            (combo_chunk, combo_document_is_valid,
                             unsuccessful_doc_to_append,
                             combo_vectorise_time_to_add) = vectorise_multimodal_combination_field_structured(
                                field_name, field_content, copied, i, doc_id, add_docs_params.device, marqo_index,
                                image_repo, mappings, model_auth=add_docs_params.model_auth)

                            total_vectorise_time = total_vectorise_time + combo_vectorise_time_to_add

                            if combo_document_is_valid is False:
                                document_is_valid = False
                                unsuccessful_docs.append(unsuccessful_doc_to_append)
                                break
                            else:
                                chunks = [combo_chunk[TensorField.field_content]]
                                embeddings = [combo_chunk[TensorField.marqo_knn_field]]

                        processed_tensor_fields[tensor_field.name] = {}
                        processed_tensor_fields[tensor_field.name][constants.MARQO_DOC_CHUNKS] = chunks
                        processed_tensor_fields[tensor_field.name][constants.MARQO_DOC_EMBEDDINGS] = embeddings

            if document_is_valid:
                if processed_tensor_fields:
                    copied[constants.MARQO_DOC_TENSORS] = processed_tensor_fields
                copied[constants.MARQO_DOC_ID] = doc_id

                try:
                    converted_doc = VespaDocument(**vespa_index.to_vespa_document(copied))
                    bulk_parent_dicts.append(converted_doc)
                except core_exceptions.MarqoDocumentParsingError as e:
                    document_is_valid = False
                    unsuccessful_docs.append(
                        (i, {'_id': doc_id, 'error': e.message,
                             'status': int(api_exceptions.InvalidArgError.status_code),
                             'code': api_exceptions.InvalidArgError.code})
                    )

    total_preproc_time = 0.001 * RequestMetricsStore.for_request().stop(
        "add_documents.processing_before_vespa")
    logger.debug(
        f"      add_documents pre-processing: took {(total_preproc_time):.3f}s total for {batch_size} docs, "
        f"for an average of {(total_preproc_time / batch_size):.3f}s per doc.")

    logger.debug(f"          add_documents vectorise: took {(total_vectorise_time):.3f}s for {batch_size} docs, "
                 f"for an average of {(total_vectorise_time / batch_size):.3f}s per doc.")

    if bulk_parent_dicts:
        # ADD DOCS TIMER-LOGGER (5)
        start_time_5 = timer()
        with RequestMetricsStore.for_request().time("add_documents.vespa._bulk"):
            index_responses = vespa_client.feed_batch(bulk_parent_dicts, marqo_index.schema_name)

        end_time_5 = timer()
        total_http_time = end_time_5 - start_time_5

        logger.debug(
            f"      add_documents roundtrip: took {(total_http_time):.3f}s to send {batch_size} docs (roundtrip) to Marqo-os, "
            f"for an average of {(total_http_time / batch_size):.3f}s per doc.")
    else:
        index_responses = None

    with RequestMetricsStore.for_request().time("add_documents.postprocess"):
        t1 = timer()

        def translate_add_doc_response(responses: Optional[FeedBatchResponse], time_diff: float) -> dict:
            """translates Vespa response dict into Marqo dict"""
            result_dict = {}
            new_items: List[Dict] = []

            if responses is not None:
                result_dict['errors'] = responses.errors

                for resp in responses.responses:
                    id = resp.id.split('::')[-1] if resp.id else None
                    new_items.append({'status': resp.status})
                    if id:
                        new_items[-1].update({'_id': id})
                    if resp.message:
                        new_items[-1].update({'message': resp.message})

            if unsuccessful_docs:
                result_dict['errors'] = True

            for loc, error_info in unsuccessful_docs:
                new_items.insert(loc, error_info)

            result_dict["processingTimeMs"] = time_diff * 1000
            result_dict["index_name"] = add_docs_params.index_name
            result_dict["items"] = new_items

            return result_dict

        return translate_add_doc_response(index_responses, time_diff=t1 - t0)


def _get_marqo_document_by_id(config: Config, index_name: str, document_id: str):
    marqo_index = index_meta_cache.get_index(config=config, index_name=index_name)

    try:
        res = config.vespa_client.get_document(document_id, marqo_index.schema_name)
    except VespaStatusError as e:
        if e.status_code == 404:
            raise api_exceptions.DocumentNotFoundError(
                f"Document with ID {document_id} not found in index {index_name}")
        else:
            raise e

    vespa_index = vespa_index_factory(marqo_index)
    marqo_document = vespa_index.to_marqo_document(res.document.dict())

    return marqo_document


def get_document_by_id(
        config: Config, index_name: str, document_id: str, show_vectors: bool = False):
    """returns document by its ID"""
    validation.validate_id(document_id)

    marqo_document = _get_marqo_document_by_id(config, index_name, document_id)

    if show_vectors:
        if constants.MARQO_DOC_TENSORS in marqo_document:
            marqo_document[TensorField.tensor_facets] = _get_tensor_facets(marqo_document[constants.MARQO_DOC_TENSORS])
        else:
            marqo_document[TensorField.tensor_facets] = []

    if not show_vectors:
        if unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS in marqo_document:
            del marqo_document[unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS]

    if constants.MARQO_DOC_TENSORS in marqo_document:
        del marqo_document[constants.MARQO_DOC_TENSORS]

    return marqo_document


def _get_marqo_documents_by_ids(
        config: Config, index_name: str, document_ids, ignore_invalid_ids: bool = False
):
    validated_ids = []
    for doc_id in document_ids:
        try:
            validated_ids.append(validation.validate_id(doc_id))
        except api_exceptions.InvalidDocumentIdError as e:
            if not ignore_invalid_ids:
                raise e
            logger.debug(f'Invalid document ID {doc_id} ignored')

    if len(validated_ids) == 0:  # Can only happen when ignore_invalid_ids is True
        return []

    marqo_index = index_meta_cache.get_index(config=config, index_name=index_name)
    batch_get = config.vespa_client.get_batch(validated_ids, marqo_index.schema_name)
    vespa_index = vespa_index_factory(marqo_index)

    return [vespa_index.to_marqo_document(response.document.dict()) for response in batch_get.responses
            if response.status == 200]


def get_documents_by_ids(
        config: Config, index_name: str, document_ids: typing.Collection[str],
        show_vectors: bool = False, ignore_invalid_ids: bool = False
) -> Dict[str, Any]:
    """
    Returns documents by their IDs.

    Args:
        ignore_invalid_ids: If True, invalid IDs will be ignored and not returned in the response. If False, an error
            will be raised if any of the IDs are invalid
    """
    if not isinstance(document_ids, typing.Collection):
        raise api_exceptions.InvalidArgError("Get documents must be passed a collection of IDs!")
    if len(document_ids) <= 0:
        raise api_exceptions.InvalidArgError("Can't get empty collection of IDs!")

    max_docs_limit = utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_RETRIEVABLE_DOCS)
    if max_docs_limit is not None and len(document_ids) > int(max_docs_limit):
        raise api_exceptions.IllegalRequestedDocCount(
            f"{len(document_ids)} documents were requested, which is more than the allowed limit of [{max_docs_limit}], "
            f"set by the environment variable `{EnvVars.MARQO_MAX_RETRIEVABLE_DOCS}`")

    validated_ids = []
    for doc_id in document_ids:
        try:
            validated_ids.append(validation.validate_id(doc_id))
        except api_exceptions.InvalidDocumentIdError as e:
            if not ignore_invalid_ids:
                raise e
            logger.debug(f'Invalid document ID {doc_id} ignored')

    if len(validated_ids) == 0:  # Can only happen when ignore_invalid_ids is True
        return {"results": []}

    marqo_index = index_meta_cache.get_index(config=config, index_name=index_name)
    batch_get = config.vespa_client.get_batch(validated_ids, marqo_index.schema_name)
    vespa_index = vespa_index_factory(marqo_index)

    to_return = {
        "results": []
    }

    for response in batch_get.responses:
        if response.status == 200:
            marqo_document = vespa_index.to_marqo_document(response.document.dict())

            if show_vectors:
                if constants.MARQO_DOC_TENSORS in marqo_document:
                    marqo_document[TensorField.tensor_facets] = _get_tensor_facets(
                        marqo_document[constants.MARQO_DOC_TENSORS])
                else:
                    marqo_document[TensorField.tensor_facets] = []

            if not show_vectors:
                if unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS in marqo_document:
                    del marqo_document[unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS]

            if constants.MARQO_DOC_TENSORS in marqo_document:
                del marqo_document[constants.MARQO_DOC_TENSORS]

            to_return['results'].append(
                {
                    TensorField.found: True,
                    **marqo_document
                }
            )
        elif response.status == 404:
            to_return['results'].append(
                {
                    TensorField.found: False,
                    '_id': _get_id_from_vespa_id(response.id)
                }
            )
        else:  # If not 200 or 404, it should have been raised by the client
            raise api_exceptions.InternalError(
                f"Unexpected response status code {response.status} for document {response.id}")

    return to_return


def _get_id_from_vespa_id(vespa_id: str) -> str:
    """Returns the document ID from a Vespa ID. Vespa IDs are of the form `namespace::document_id`."""
    return vespa_id.split('::')[-1]


def _get_tensor_facets(marqo_doc_tensors: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Reformat Marqo doc tensors field for API response
    """
    tensor_facets = []
    for tensor_field in marqo_doc_tensors:
        chunks = marqo_doc_tensors[tensor_field][constants.MARQO_DOC_CHUNKS]
        embeddings = marqo_doc_tensors[tensor_field][constants.MARQO_DOC_EMBEDDINGS]
        if len(chunks) != len(embeddings):
            raise api_exceptions.InternalError(
                f"Number of chunks ({len(chunks)}) and number of embeddings ({len(embeddings)}) "
                f"for field {tensor_field} must be the same.")

        for i in range(len(chunks)):
            tensor_facets.append(
                {
                    tensor_field: chunks[i],
                    TensorField.embedding: embeddings[i]
                }
            )

    return tensor_facets


def rerank_query(query: BulkSearchQueryEntity, result: Dict[str, Any], reranker: Union[str, Dict], device: str,
                 num_highlights: int):
    if query.searchableAttributes is None:
        raise api_exceptions.InvalidArgError(
            f"searchable_attributes cannot be None when re-ranking. Specify which fields to search and rerank over.")
    try:
        start_rerank_time = timer()
        rerank.rerank_search_results(search_result=result, query=query.q,
                                     model_name=reranker, device=device,
                                     searchable_attributes=query.searchableAttributes, num_highlights=num_highlights)
        logger.debug(
            f"search ({query.searchMethod.lower()}) reranking using {reranker}: took {(timer() - start_rerank_time):.3f}s to rerank results.")
    except Exception as e:
        raise api_exceptions.BadRequestError(f"reranking failure due to {str(e)}")


def search(config: Config, index_name: str, text: Union[str, dict],
           result_count: int = 3, offset: int = 0,
           highlights: bool = True, ef_search: Optional[int] = None,
           approximate: Optional[bool] = None,
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
    if result_count <= 0 or (not isinstance(result_count, int)):
        raise errors.IllegalRequestedDocCount(
            f"result_count must be an integer greater than 0! Received {result_count}"
        )

    if offset < 0:
        raise api_exceptions.IllegalRequestedDocCount("search result offset cannot be less than 0!")

        # validate query
    validation.validate_query(q=text, search_method=search_method)

    # Validate max limits
    max_docs_limit = utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_RETRIEVABLE_DOCS)
    max_search_limit = utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_SEARCH_LIMIT)
    max_search_offset = utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_SEARCH_OFFSET)

    check_upper = True if max_docs_limit is None else result_count + offset <= int(max_docs_limit)
    check_limit = True if max_search_limit is None else result_count <= int(max_search_limit)
    check_offset = True if max_search_offset is None else offset <= int(max_search_offset)
    if not check_upper:
        upper_bound_explanation = ("The search result limit + offset must be less than or equal to the "
                                   f"MARQO_MAX_RETRIEVABLE_DOCS limit of [{max_docs_limit}]. ")

        raise api_exceptions.IllegalRequestedDocCount(
            f"{upper_bound_explanation} Marqo received search result limit of `{result_count}` "
            f"and offset of `{offset}`.")
    if not check_limit:
        raise api_exceptions.IllegalRequestedDocCount(
            f"The search result limit must be less than or equal to the MARQO_MAX_SEARCH_LIMIT limit of "
            f"[{max_search_limit}]. Marqo received search result limit of `{result_count}`.")
    if not check_offset:
        raise api_exceptions.IllegalRequestedDocCount(
            f"The search result offset must be less than or equal to the MARQO_MAX_SEARCH_OFFSET limit of "
            f"[{max_search_offset}]. Marqo received search result offset of `{offset}`.")

    t0 = timer()
    validation.validate_context(context=context, query=text, search_method=search_method)
    validation.validate_boost(boost=boost, search_method=search_method)
    validation.validate_searchable_attributes(searchable_attributes=searchable_attributes, search_method=search_method)
    if searchable_attributes is not None:
        [validation.validate_field_name(attribute) for attribute in searchable_attributes]
    if attributes_to_retrieve is not None:
        if not isinstance(attributes_to_retrieve, (List, typing.Tuple)):
            raise api_exceptions.InvalidArgError("attributes_to_retrieve must be a sequence!")
        [validation.validate_field_name(attribute) for attribute in attributes_to_retrieve]
    if verbose:
        print(f"determined_search_method: {search_method}, text query: {text}")

    if device is None:
        selected_device = utils.read_env_vars_and_defaults("MARQO_BEST_AVAILABLE_DEVICE")
        if selected_device is None:
            raise api_exceptions.InternalError("Best available device was not properly determined on Marqo startup.")
        logger.debug(f"No device given for search. Defaulting to best available device: {selected_device}")
    else:
        selected_device = device

    if search_method.upper() == SearchMethod.TENSOR:
        # Default approximate and efSearch -- we can't set these at API-level since they're not a valid args
        # for lexical search
        if ef_search is None:
            # efSearch must be min result_count + offset
            ef_search = max(
                utils.read_env_vars_and_defaults_ints(EnvVars.MARQO_DEFAULT_EF_SEARCH),
                result_count + offset
            )
        if approximate is None:
            approximate = True

        search_result = _vector_text_search(
            config=config, index_name=index_name, query=text, result_count=result_count, offset=offset,
            ef_search=ef_search, approximate=approximate, searchable_attributes=searchable_attributes,
            filter_string=filter, device=selected_device, attributes_to_retrieve=attributes_to_retrieve, boost=boost,
            image_download_headers=image_download_headers, context=context, score_modifiers=score_modifiers,
            model_auth=model_auth, highlights=highlights
        )
    elif search_method.upper() == SearchMethod.LEXICAL:
        if ef_search is not None:
            raise errors.InvalidArgError(
                f"efSearch is not a valid argument for lexical search")
        if approximate is not None:
            raise errors.InvalidArgError(
                f"approximate is not a valid argument for lexical search")
        if score_modifiers is not None:
            raise errors.InvalidArgError(
                "Score modifiers is not supported for lexical search yet"
            )

        search_result = _lexical_search(
            config=config, index_name=index_name, text=text, result_count=result_count, offset=offset,
            searchable_attributes=searchable_attributes, verbose=verbose,
            filter_string=filter, attributes_to_retrieve=attributes_to_retrieve, highlights=highlights
        )
    else:
        raise api_exceptions.InvalidArgError(f"Search called with unknown search method: {search_method}")

    if reranker is not None:
        logger.info("reranking using {}".format(reranker))
        if searchable_attributes is None:
            raise api_exceptions.InvalidArgError(
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
            raise api_exceptions.BadRequestError(f"reranking failure due to {str(e)}")

    search_result["query"] = text
    search_result["limit"] = result_count
    search_result["offset"] = offset

    time_taken = timer() - t0
    search_result["processingTimeMs"] = round(time_taken * 1000)
    logger.debug(f"search ({search_method.lower()}) completed with total processing time: {(time_taken):.3f}s.")

    return search_result


def _lexical_search(
        config: Config, index_name: str, text: str, result_count: int = 3, offset: int = 0,
        searchable_attributes: Sequence[str] = None, verbose: int = 0, filter_string: str = None,
        highlights: bool = True, attributes_to_retrieve: Optional[List[str]] = None, expose_facets: bool = False):
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
        raise api_exceptions.InvalidArgError(
            f"Query arg must be of type str! text arg is of type {type(text)}. "
            f"Query arg: {text}")

    marqo_index = index_meta_cache.get_index(config=config, index_name=index_name)

    # SEARCH TIMER-LOGGER (pre-processing)
    RequestMetricsStore.for_request().start("search.lexical.processing_before_vespa")

    # Parse text into required and optional terms.
    (required_terms, optional_terms) = utils.parse_lexical_query(text)

    marqo_query = MarqoLexicalQuery(
        index_name=index_name,
        or_phrases=optional_terms,
        and_phrases=required_terms,
        filter=filter_string,
        limit=result_count,
        offset=offset,
        searchable_attributes=searchable_attributes,
        attributes_to_retrieve=attributes_to_retrieve,
    )

    vespa_index = vespa_index_factory(marqo_index)
    vespa_query = vespa_index.to_vespa_query(marqo_query)

    total_preprocess_time = RequestMetricsStore.for_request().stop("search.lexical.processing_before_vespa")
    logger.debug(f"search (lexical) pre-processing: took {(total_preprocess_time):.3f}ms to process query.")

    with RequestMetricsStore.for_request().time("search.lexical.vespa",
                                                lambda t: logger.debug(f"Vespa search: took {t:.3f}ms")
                                                ):
        responses = config.vespa_client.query(**vespa_query)

    # SEARCH TIMER-LOGGER (post-processing)
    RequestMetricsStore.for_request().start("search.lexical.postprocess")
    gathered_docs = gather_documents_from_response(responses, marqo_index, False, attributes_to_retrieve)

    # Set the _highlights for each doc as [] to follow Marqo-V1's convention
    if highlights:
        for docs in gathered_docs['hits']:
            docs['_highlights'] = []

    total_postprocess_time = RequestMetricsStore.for_request().stop("search.lexical.postprocess")
    logger.debug(
        f"search (lexical) post-processing: took {(total_postprocess_time):.3f}ms to format "
        f"{len(gathered_docs)} results."
    )

    return gathered_docs


def construct_vector_input_batches(query: Union[str, Dict], index_info: MarqoIndex) -> Tuple[List[str], List[str]]:
    """Splits images from text in a single query (either a query string, or dict of weighted strings).

    Args:
        query: a string query, or a dict of weighted strings.
        index_info: used to determine whether URLs should be treated as images

    Returns:
        A tuple of string batches. The first is text content the second is image content.
    """
    # TODO - infer this from model
    treat_urls_as_images = True
    if isinstance(query, str):
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


def gather_documents_from_response(response: QueryResult, marqo_index: MarqoIndex, highlights: bool,
                                   attributes_to_retrieve: List[str] = None) -> Dict[str, Any]:
    """
    Convert a VespaQueryResponse to a Marqo search response
    """
    vespa_index = vespa_index_factory(marqo_index)
    hits = []
    for doc in response.hits:
        marqo_doc = vespa_index.to_marqo_document(doc.dict(), return_highlights=highlights)
        marqo_doc['_score'] = doc.relevance

        if marqo_index.type == IndexType.Unstructured and attributes_to_retrieve is not None:
            # For an unstructured index, we do the attributes_to_retrieve after search
            marqo_doc = unstructured_index_attributes_to_retrieve(marqo_doc, attributes_to_retrieve)

        # Delete chunk data
        if constants.MARQO_DOC_TENSORS in marqo_doc:
            del marqo_doc[constants.MARQO_DOC_TENSORS]
        hits.append(marqo_doc)

    return {'hits': hits}


def unstructured_index_attributes_to_retrieve(marqo_doc: Dict[str, Any], attributes_to_retrieve: List[str]) -> Dict[
    str, Any]:
    # attributes_to_retrieve should already be validated at the start of search
    attributes_to_retrieve = list(set(attributes_to_retrieve).union({"_id", "_score", "_highlights"}))
    return {k: v for k, v in marqo_doc.items() if k in attributes_to_retrieve}


def assign_query_to_vector_job(
        q: BulkSearchQueryEntity, jobs: Dict[JHash, VectorisedJobs], grouped_content: Tuple[List[str], List[str]],
        index_info: MarqoIndex, device: str) -> List[VectorisedJobPointer]:
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
            model_name=index_info.model.name,
            model_properties=index_info.model.get_properties(),
            content=grouped_content,
            device=device,
            normalize_embeddings=index_info.normalize_embeddings,
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


def create_vector_jobs(queries: List[BulkSearchQueryEntity], config: Config, device: str) -> Tuple[
    Dict[Qidx, List[VectorisedJobPointer]], Dict[JHash, VectorisedJobs]]:
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
        index_info = get_index(config=config, index_name=q.index)
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
            raise api_exceptions.BadRequestError(
                message=f'Problem vectorising query. Reason: {str(model_error)}',
                link=marqo_docs.list_of_models()
            )

        except s2_inference_errors.S2InferenceError as e:
            # TODO: differentiate image processing errors from other types of vectorise errors
            raise api_exceptions.InvalidArgError(message=f'Error vectorising content: {v.content}. Message: {e}')
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

        # vectors = job_to_vectors[ptrs.job_hash][ptrs.start_idx: ptrs.end_idx]

        # qidx_to_vectors[qidx].append(vectors)
        q = queries[qidx]
        index_info = get_index(config=config, index_name=q.index)

        ordered_queries = list(q.q.items()) if isinstance(q.q, dict) else None
        if ordered_queries:
            # multiple queries. We have to weight and combine them:
            vectorised_ordered_queries = [
                (get_content_vector(
                    possible_jobs=qidx_to_job[qidx],
                    jobs=jobs,
                    job_to_vectors=job_to_vectors,
                    treat_urls_as_images=True,  # TODO - infer this from model
                    content=content),
                 weight,
                 content
                ) for content, weight in ordered_queries
            ]
            # TODO how do we ensure order?
            weighted_vectors = [np.asarray(vec) * weight for vec, weight, content in vectorised_ordered_queries]

            context_tensors = q.get_context_tensor()
            if context_tensors is not None:
                weighted_vectors += [np.asarray(v.vector) * v.weight for v in context_tensors]

            try:
                merged_vector = np.mean(weighted_vectors, axis=0)
            except ValueError as e:
                raise api_exceptions.InvalidArgError(f"The provided vectors are not in the same dimension of the index."
                                                     f"This causes the error when we do `numpy.mean()` over all the vectors.\n"
                                                     f"The original error is `{e}`.\n"
                                                     f"Please check `{marqo_docs.search_context()}`.")

            if index_info.normalize_embeddings:
                norm = np.linalg.norm(merged_vector, axis=-1, keepdims=True)
                if norm > 0:
                    merged_vector /= np.linalg.norm(merged_vector, axis=-1, keepdims=True)
            result[qidx] = list(merged_vector)
        else:
            # result[qidx] = vectors[0]
            result[qidx] = get_content_vector(
                possible_jobs=qidx_to_job.get(qidx, []),
                jobs=jobs,
                job_to_vectors=job_to_vectors,
                treat_urls_as_images=True,  # TODO - infer this from model
                content=q.q
            )
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


def run_vectorise_pipeline(config: Config, queries: List[BulkSearchQueryEntity], device: Union[Device, str]) -> Dict[
    Qidx, List[float]]:
    """Run the query vectorisation process"""
    # 1. Pre-process inputs ready for s2_inference.vectorise
    # we can still use qidx_to_job. But the jobs structure may need to be different
    vector_jobs_tuple: Tuple[Dict[Qidx, List[VectorisedJobPointer]], Dict[JHash, VectorisedJobs]] = create_vector_jobs(
        queries, config, device)

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


def _vector_text_search(
        config: Config, index_name: str, query: Union[str, dict], result_count: int = 5, offset: int = 0,
        ef_search: Optional[int] = None, approximate: bool = True,
        searchable_attributes: Iterable[str] = None, filter_string: str = None, device: str = None,
        attributes_to_retrieve: Optional[List[str]] = None, boost: Optional[Dict] = None,
        image_download_headers: Optional[Dict] = None, context: Optional[Dict] = None,
        score_modifiers: Optional[ScoreModifier] = None, model_auth: Optional[ModelAuth] = None,
        highlights: bool = False) -> Dict:
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
        highlights: if True, highlights will be returned
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
        raise api_exceptions.InternalError("_vector_text_search cannot be called without `device`!")

    RequestMetricsStore.for_request().start("search.vector.processing_before_vespa")

    marqo_index = index_meta_cache.get_index(config=config, index_name=index_name)

    queries = [BulkSearchQueryEntity(
        q=query, searchableAttributes=searchable_attributes, searchMethod=SearchMethod.TENSOR, limit=result_count,
        offset=offset, showHighlights=False, filter=filter_string, attributesToRetrieve=attributes_to_retrieve,
        boost=boost, image_download_headers=image_download_headers, context=context, scoreModifiers=score_modifiers,
        index=index_name, modelAuth=model_auth
    )]
    with RequestMetricsStore.for_request().time(f"search.vector_inference_full_pipeline"):
        qidx_to_vectors: Dict[Qidx, List[float]] = run_vectorise_pipeline(config, queries, device)
    vectorised_text = list(qidx_to_vectors.values())[0]

    marqo_query = MarqoTensorQuery(
        index_name=index_name,
        vector_query=vectorised_text,
        filter=filter_string,
        limit=result_count,
        ef_search=ef_search,
        approximate=approximate,
        offset=offset,
        searchable_attributes=searchable_attributes,
        attributes_to_retrieve=attributes_to_retrieve,
        score_modifiers=score_modifiers.to_marqo_score_modifiers() if score_modifiers is not None else None
    )

    vespa_index = vespa_index_factory(marqo_index)
    vespa_query = vespa_index.to_vespa_query(marqo_query)

    total_preprocess_time = RequestMetricsStore.for_request().stop("search.vector.processing_before_vespa")
    logger.debug(
        f"search (tensor) pre-processing: took {(total_preprocess_time):.3f}ms to vectorize and process query.")

    # SEARCH TIMER-LOGGER (roundtrip)
    with RequestMetricsStore.for_request().time("search.vector.vespa",
                                                lambda t: logger.debug(f"Vespa search: took {t:.3f}ms")
                                                ):
        responses = config.vespa_client.query(**vespa_query)

    if not approximate and (responses.root.coverage.coverage < 100 or responses.root.coverage.degraded is not None):
        raise errors.InternalError(
            f'Graceful degradation detected for non-approximate search. '
            f'Coverage is not 100%: {responses.root.coverage}'
            f'Degraded: {str(responses.root.coverage.degraded)}'
        )

    # SEARCH TIMER-LOGGER (post-processing)
    RequestMetricsStore.for_request().start("search.vector.postprocess")
    gathered_docs = gather_documents_from_response(responses, marqo_index, highlights, attributes_to_retrieve)

    if boost is not None:
        raise api_exceptions.MarqoWebError('Boosting is not currently supported with Vespa')
        # gathered_docs = boost_score(gathered_docs, boost, searchable_attributes)

    # completely_sorted = sort_chunks(gathered_docs)

    # if verbose:
    #     print("Chunk vector search, sorted result:")
    #     if verbose == 1:
    #         pprint.pprint(utils.truncate_dict_vectors(completely_sorted))
    #     elif verbose == 2:
    #         pprint.pprint(completely_sorted)

    total_postprocess_time = RequestMetricsStore.for_request().stop("search.vector.postprocess")
    logger.debug(
        f"search (tensor) post-processing: took {(total_postprocess_time):.3f}ms to sort and format "
        f"{len(gathered_docs)} results from Vespa."
    )

    return gathered_docs


def delete_index(config: Config, index_name):
    index_management = IndexManagement(vespa_client=config.vespa_client)
    index_management.delete_index_by_name(index_name)

    if index_name in get_cache():
        del get_cache()[index_name]


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
        raise api_exceptions.ModelNotInCacheError(message=str(e))
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
        raise api_exceptions.HardwareCompatabilityError(message=str(
            "ERROR: cuda is not supported in your machine!!"
        ))


def vectorise_multimodal_combination_field_unstructured(field: str,
                                                        field_content: Dict[str, str], doc_index: int,
                                                        doc_id: str, device: str, marqo_index: UnstructuredMarqoIndex,
                                                        image_repo, field_map: dict,
                                                        model_auth: Optional[ModelAuth] = None
                                                        ):
    '''
    This function is used to vectorise multimodal combination field.
    Over all this is a simplified version of the vectorise pipeline in add_documents. Specifically,
    1. we don't do any chunking here.
    2. we don't use image repo for concurrent downloading.
    Args:
        field_name: the name of the multimodal
        field_content: the subfields name and content, e.g.,
            {"subfield_one" : "content-1",
             "subfield_two" : "content-2"},
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
    combo_document_is_valid = True
    combo_vectorise_time_to_add = 0
    combo_chunk = {}
    combo_embeddings = []
    unsuccessful_doc_to_append = tuple()
    new_fields_from_multimodal_combination = set()

    # Copy the important mutable objects from main body for safety purpose
    field_content_copy = copy.deepcopy(field_content)

    # 4 lists to store the field name and field content to vectorise.
    text_field_names = []
    text_content_to_vectorise = []

    image_field_names = []
    image_content_to_vectorise = []

    normalize_embeddings = marqo_index.normalize_embeddings
    infer_if_image = marqo_index.treat_urls_and_pointers_as_images

    if infer_if_image is False:
        text_field_names = list(field_content.keys())
        text_content_to_vectorise = list(field_content.values())
    else:
        for sub_field_name, sub_content in field_content.items():
            if isinstance(sub_content, str) and not _is_image(sub_content):
                text_field_names.append(sub_field_name)
                text_content_to_vectorise.append(sub_content)
            else:
                try:
                    if isinstance(sub_content, str) and marqo_index.treat_urls_and_pointers_as_images:
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

                    return combo_chunk, combo_embeddings, combo_document_is_valid, unsuccessful_doc_to_append, combo_vectorise_time_to_add

    try:
        start_time = timer()
        text_vectors = []
        if len(text_content_to_vectorise) > 0:
            with RequestMetricsStore.for_request().time(f"create_vectors"):
                text_vectors = s2_inference.vectorise(
                    model_name=marqo_index.model.name,
                    model_properties=marqo_index.model.properties, content=text_content_to_vectorise,
                    device=device, normalize_embeddings=normalize_embeddings,
                    infer=infer_if_image, model_auth=model_auth
                )
        image_vectors = []
        if len(image_content_to_vectorise) > 0:
            with RequestMetricsStore.for_request().time(f"create_vectors"):
                image_vectors = s2_inference.vectorise(
                    model_name=marqo_index.model.name,
                    model_properties=marqo_index.model.properties, content=image_content_to_vectorise,
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
            link=marqo_docs.list_of_models()
        )
    except s2_inference_errors.S2InferenceError:
        combo_document_is_valid = False
        image_err = errors.InvalidArgError(message=f'Could not process given image: {field_content_copy}')
        unsuccessful_doc_to_append = \
            (doc_index, {'_id': doc_id, 'error': image_err.message, 'status': int(image_err.status_code),
                         'code': image_err.code})

        return combo_chunk, combo_embeddings, combo_document_is_valid, unsuccessful_doc_to_append, combo_vectorise_time_to_add

    sub_field_name_list = text_field_names + image_field_names
    vectors_list = text_vectors + image_vectors

    if not len(sub_field_name_list) == len(vectors_list):
        raise errors.BatchInferenceSizeError(
            message=f"Batch inference size does not match content for multimodal field {field}")

    vector_chunk = np.squeeze(np.mean(
        [np.array(vector) * field_map["weights"][sub_field_name] for sub_field_name, vector in
         zip(sub_field_name_list, vectors_list)], axis=0))

    if normalize_embeddings is True:
        vector_chunk = vector_chunk / np.linalg.norm(vector_chunk)

    combo_embeddings: List[float] = vector_chunk.tolist()
    combo_chunk: str = f"{field}::{json.dumps(field_content)}"

    return combo_chunk, combo_embeddings, combo_document_is_valid, unsuccessful_doc_to_append, combo_vectorise_time_to_add


def vectorise_multimodal_combination_field_structured(
        field: str, multimodal_object: Dict[str, dict], doc: dict, doc_index: int,
        doc_id: str, device: str, marqo_index: StructuredMarqoIndex, image_repo, field_map: dict,
        model_auth: Optional[ModelAuth] = None
):
    """
    This function is used to vectorise multimodal combination field. The field content should
    have the following structure:
    field_content = {"tensor_field_one" : {"weight":0.5, "parameter": "test-parameter-1"},
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
        marqo_index: index_info from main body,
        model_auth: Model download authorisation information (if required)
    Returns:
        combo_chunk: the combo_chunk to be appended to the main body
        combo_document_is_valid:  if the document is a valid
        unsuccessful_docs: appended unsucessful_docs
        combo_total_vectorise_time: the vectorise time spent in combo field

    """
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

    normalize_embeddings = marqo_index.normalize_embeddings
    image_fields = [field.name for field in marqo_index.field_map_by_type[FieldType.ImagePointer]]

    for sub_field_name, sub_content in multimodal_object.items():
        if isinstance(sub_content, str) and sub_field_name not in image_fields:
            text_field_names.append(sub_field_name)
            text_content_to_vectorise.append(sub_content)
        else:
            try:
                if isinstance(sub_content, str):
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
                                 'status': int(api_exceptions.InvalidArgError.status_code),
                                 'code': api_exceptions.InvalidArgError.code})

                return combo_chunk, combo_document_is_valid, unsuccessful_doc_to_append, combo_vectorise_time_to_add

    try:
        start_time = timer()
        text_vectors = []
        if len(text_content_to_vectorise) > 0:
            with RequestMetricsStore.for_request().time(f"create_vectors"):
                text_vectors = s2_inference.vectorise(
                    model_name=marqo_index.model.name,
                    model_properties=marqo_index.model.get_properties(), content=text_content_to_vectorise,
                    device=device, normalize_embeddings=normalize_embeddings,
                    infer=False, model_auth=model_auth
                )
        image_vectors = []
        if len(image_content_to_vectorise) > 0:
            with RequestMetricsStore.for_request().time(f"create_vectors"):
                image_vectors = s2_inference.vectorise(
                    model_name=marqo_index.model.name,
                    model_properties=marqo_index.model.get_properties(), content=image_content_to_vectorise,
                    device=device, normalize_embeddings=normalize_embeddings,
                    infer=True, model_auth=model_auth
                )
        end_time = timer()
        combo_vectorise_time_to_add += (end_time - start_time)
    except (s2_inference_errors.UnknownModelError,
            s2_inference_errors.InvalidModelPropertiesError,
            s2_inference_errors.ModelLoadError) as model_error:
        raise api_exceptions.BadRequestError(
            message=f'Problem vectorising query. Reason: {str(model_error)}',
            link=marqo_docs.list_of_models()
        )
    except s2_inference_errors.S2InferenceError:
        combo_document_is_valid = False
        image_err = api_exceptions.InvalidArgError(message=f'Could not process given image: {multimodal_object_copy}')
        unsuccessful_doc_to_append = \
            (doc_index, {'_id': doc_id, 'error': image_err.message, 'status': int(image_err.status_code),
                         'code': image_err.code})

        return combo_chunk, combo_document_is_valid, unsuccessful_doc_to_append, combo_vectorise_time_to_add

    sub_field_name_list = text_field_names + image_field_names
    vectors_list = text_vectors + image_vectors

    if not len(sub_field_name_list) == len(vectors_list):
        raise api_exceptions.BatchInferenceSizeError(
            message=f"Batch inference size does not match content for multimodal field {field}")

    vector_chunk = np.squeeze(np.mean(
        [np.array(vector) * field_map["weights"][sub_field_name] for sub_field_name, vector in
         zip(sub_field_name_list, vectors_list)], axis=0))

    if normalize_embeddings is True:
        vector_chunk = vector_chunk / np.linalg.norm(vector_chunk)

    vector_chunk = vector_chunk.tolist()

    combo_chunk = dict({
        TensorField.marqo_knn_field: vector_chunk,
        TensorField.field_content: json.dumps(multimodal_object),
        TensorField.field_name: field,
    })
    return combo_chunk, combo_document_is_valid, unsuccessful_doc_to_append, combo_vectorise_time_to_add


def delete_documents(config: Config, index_name: str, doc_ids: List[str]):
    """Delete documents from the Marqo index with the given doc_ids """
    # Make sure the index exists
    marqo_index = index_meta_cache.get_index(config=config, index_name=index_name)

    return delete_docs.delete_documents(
        config=config,
        del_request=MqDeleteDocsRequest(
            index_name=index_name,
            schema_name=marqo_index.schema_name,
            document_ids=doc_ids,
        )
    )