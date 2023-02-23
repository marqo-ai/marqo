"""tensor search logic. In the future this will be accessible to the client via an API

API Notes:
    - Fields beginning with a double underscore "__" are protected and used for our internal purposes.
    - Examples include:
        __embedding_vector
        __field_name
        __field_content
        __doc_chunk_relation
        __chunk_ids
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
import datetime
from timeit import default_timer as timer
import functools
import pprint
import typing
import uuid
from typing import List, Optional, Union, Iterable, Sequence, Dict, Any
import numpy as np
from PIL import Image
from marqo.tensor_search.enums import (
    MediaType, MlModel, TensorField, SearchMethod, OpenSearchDataType,
    EnvVars
)
from marqo.tensor_search.enums import IndexSettingsField as NsField
from marqo.tensor_search import utils, backend, validation, configs, parallel, add_docs
from marqo.tensor_search.formatting import _clean_doc
from marqo.tensor_search.index_meta_cache import get_cache, get_index_info
from marqo.tensor_search import index_meta_cache
from marqo.tensor_search.models.index_info import IndexInfo
from marqo.tensor_search import constants
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

from marqo.tensor_search.tensor_search_logging import get_logger

logger = get_logger(__name__)


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

    response = HttpRequests(config).put(path=index_name, body=vector_index_settings)

    get_cache()[index_name] = IndexInfo(
        model_name=model_name, properties=vector_index_settings["mappings"]["properties"].copy(),
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

    Each Marqo field generates 3 Marqo-OS fields:
        - One for its content
        - One for its vector
        - One for filtering

    There are also 3 fields that will be generated on a Marqo index, in most
    cases:
        - one for the chunks field
        - one for chunk's __field_content
        - one for chunk's __field_name

    Returns:
        The corresponding Marqo-OS limit
    """
    return (marqo_index_field_limit * 3) + 3


def _autofill_index_settings(index_settings: dict):
    """A half-complete index settings will be auto filled"""

    # TODO: validated conflicting settings
    # treat_urls_and_pointers_as_images

    copied_settings = index_settings.copy()
    default_settings = configs.get_default_index_settings()

    copied_settings = utils.merge_dicts(default_settings, copied_settings)

    # if NsField.index_defaults not in copied_settings:
    #     copied_settings[NsField.index_defaults] = default_settings[NsField.index_defaults]

    if NsField.treat_urls_and_pointers_as_images in copied_settings[NsField.index_defaults] and \
            copied_settings[NsField.index_defaults][NsField.treat_urls_and_pointers_as_images] is True \
            and copied_settings[NsField.index_defaults][NsField.model] is None:
        copied_settings[NsField.index_defaults][NsField.model] = MlModel.clip

    # make sure the first level of keys are present, if not add all of those defaults
    # for key in list(default_settings):
    #     if key not in copied_settings or copied_settings[key] is None:
    #         copied_settings[key] = default_settings[key]

    # # make sure the first level of keys in index defaults is present, if not add all of those defaults
    # for key in list(default_settings[NsField.index_defaults]):
    #     if key not in copied_settings[NsField.index_defaults] or \
    #             copied_settings[NsField.index_defaults][key] is None:
    #         copied_settings[NsField.index_defaults][key] = default_settings[NsField.index_defaults][key]

    # text preprocessing sub fields - fills any missing sub-dict fields if some of the first level are present
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
    doc_count = HttpRequests(config).post(path=F"{index_name}/_count")["count"]
    return {
        "numberOfDocuments": doc_count
    }


def _check_and_create_index_if_not_exist(config: Config, index_name: str):
    try:
        index_info = backend.get_index_info(config=config, index_name=index_name)
    except errors.IndexNotFoundError as s:
        create_vector_index(config=config, index_name=index_name)
        index_info = backend.get_index_info(config=config, index_name=index_name)


def add_documents_orchestrator(
        config: Config, index_name: str, docs: List[dict],
        auto_refresh: bool, batch_size: int = 0, processes: int = 1,
        non_tensor_fields=None, image_download_headers: dict = None,
        device=None, update_mode: str = 'replace', use_existing_tensors: bool = False):
    if image_download_headers is None:
        image_download_headers = dict()

    if non_tensor_fields is None:
        non_tensor_fields = []

    if batch_size is None or batch_size == 0:
        logger.info(f"batch_size={batch_size} and processes={processes} - not doing any marqo side batching")
        return add_documents(
            config=config, index_name=index_name, docs=docs, auto_refresh=auto_refresh,
            device=device, update_mode=update_mode, non_tensor_fields=non_tensor_fields,
            use_existing_tensors=use_existing_tensors, image_download_headers=image_download_headers
        )
    elif processes is not None and processes > 1:

        # create beforehand or pull from the cache so it is upto date for the multi-processing
        _check_and_create_index_if_not_exist(config=config, index_name=index_name)

        logger.info(f"batch_size={batch_size} and processes={processes} - using multi-processing")
        results = parallel.add_documents_mp(
            config=config, index_name=index_name, docs=docs,
            auto_refresh=auto_refresh, batch_size=batch_size, processes=processes,
            device=device, update_mode=update_mode, non_tensor_fields=non_tensor_fields,
            use_existing_tensors=use_existing_tensors, image_download_headers=image_download_headers
        )

        # we need to force the cache to update as it does not propagate using mp
        # we just clear this index's entry and it will re-populate when needed next
        if index_name in get_cache():
            logger.info(f'deleting cache entry for {index_name} after parallel add documents')
            del get_cache()[index_name]

        return results
    else:
        if batch_size < 0:
            raise errors.InvalidArgError("Batch size can't be less than 1!")
        logger.info(f"batch_size={batch_size} and processes={processes} - batching using a single process")
        return _batch_request(config=config, index_name=index_name, dataset=docs, device=device,
                              batch_size=batch_size, verbose=False, non_tensor_fields=non_tensor_fields,
                              use_existing_tensors=use_existing_tensors,
                              image_download_headers=image_download_headers)


def _batch_request(config: Config, index_name: str, dataset: List[dict],
                   batch_size: int = 100, verbose: bool = True, device=None,
                   update_mode: str = 'replace', non_tensor_fields=None,
                   image_download_headers: Optional[Dict] = None, use_existing_tensors: bool = False
                   ) -> List[Dict[str, Any]]:
    """Batch by the number of documents"""
    if image_download_headers is None:
        image_download_headers = dict()

    if non_tensor_fields is None:
        non_tensor_fields = []

    logger.info(f"starting batch ingestion in sizes of {batch_size}")

    deeper = ((doc, i, batch_size) for i, doc in enumerate(dataset))

    def batch_requests(gathered, doc_tuple):
        doc, i, the_batch_size = doc_tuple
        if i % the_batch_size == 0:
            gathered.append([doc, ])
        else:
            gathered[-1].append(doc)
        return gathered

    batched = functools.reduce(lambda x, y: batch_requests(x, y), deeper, [])

    def verbosely_add_docs(i, docs):
        t0 = timer()

        logger.info(f"    batch {i}: beginning ingestion. ")
        res = add_documents(
            config=config, index_name=index_name,
            docs=docs, auto_refresh=False, device=device,
            update_mode=update_mode, non_tensor_fields=non_tensor_fields,
            use_existing_tensors=use_existing_tensors, image_download_headers=image_download_headers
        )
        total_batch_time = timer() - t0
        num_docs = len(docs)

        logger.info(f"    batch {i}: ingested {num_docs} docs. Time taken: {(total_batch_time):.3f}. "
                    f"Average time per doc {(total_batch_time / num_docs):.3f}")
        if verbose:
            logger.info(f"        results from indexing batch {i}: {res}")
        return res

    results = [verbosely_add_docs(i, docs) for i, docs in enumerate(batched)]
    logger.info('completed batch ingestion.')
    return results


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
    elif isinstance(sample_field_content, dict):
        to_check = list(sample_field_content.keys())[0]
    else:
        to_check = sample_field_content

    if isinstance(to_check, str):
        return OpenSearchDataType.text
    else:
        return None


def _get_chunks_for_field(field_name: str, doc_id: str, doc):
    # Find the chunks with a specific __field_name in a doc
    return [chunk for chunk in doc["_source"]["__chunks"] if chunk["__field_name"] == field_name]


def add_documents(config: Config, index_name: str, docs: List[dict], auto_refresh: bool,
                  non_tensor_fields=None, device=None, update_mode: str = "replace",
                  image_download_thread_count: int = 20, image_download_headers: dict = None,
                  use_existing_tensors: bool = False):
    """

    Args:
        config: Config object
        index_name: name of the index
        docs: List of documents
        auto_refresh: Set to False if indexing lots of docs
        non_tensor_fields: List of fields, within documents to not create tensors for. Default to
          make tensors for all fields.
        use_existing_tensors: Whether or not to use the vectors already in doc (for update docs)
        device: Device used to carry out the document update.
        update_mode: {'replace' | 'update'}. If set to replace (default) just
        image_download_thread_count: number of threads used to concurrently download images
        image_download_headers: headers to authenticate image download
    Returns:

    """
    # ADD DOCS TIMER-LOGGER (3)
    if image_download_headers is None:
        image_download_headers = dict()
    start_time_3 = timer()

    if non_tensor_fields is None:
        non_tensor_fields = []

    t0 = timer()
    bulk_parent_dicts = []

    try:
        index_info = backend.get_index_info(config=config, index_name=index_name)
    except errors.IndexNotFoundError as s:
        create_vector_index(config=config, index_name=index_name)
        index_info = backend.get_index_info(config=config, index_name=index_name)

    if len(docs) == 0:
        raise errors.BadRequestError(message="Received empty add documents request")

    if use_existing_tensors and update_mode != "replace":
        raise errors.InvalidArgError("use_existing_tensors=True is only available for add and replace documents,"
                                     "not for add and update!")

    valid_update_modes = ('update', 'replace')
    if update_mode not in valid_update_modes:
        raise errors.InvalidArgError(message=f"Unknown update_mode `{update_mode}` "
                                             f"received! Valid update modes: {valid_update_modes}")

    existing_fields = set(index_info.properties.keys())
    new_fields = set()

    selected_device = config.indexing_device if device is None else device

    unsuccessful_docs = []
    total_vectorise_time = 0
    batch_size = len(docs)
    image_repo = {}

    if index_info.index_settings[NsField.index_defaults][NsField.treat_urls_and_pointers_as_images]:
        ti_0 = timer()
        image_repo = add_docs.download_images(docs=docs, thread_count=20, non_tensor_fields=tuple(non_tensor_fields),
                                              image_download_headers=image_download_headers)
        logger.info(f"          add_documents image download: took {(timer() - ti_0):.3f}s to concurrently download "
                    f"images for {batch_size} docs using {image_download_thread_count} threads ")

    if update_mode == 'replace' and use_existing_tensors:
        # Get existing documents
        doc_ids = [doc["_id"] for doc in docs if "_id" in doc]
        existing_docs = _get_documents_for_upsert(config=config, index_name=index_name, document_ids=doc_ids)

    for i, doc in enumerate(docs):

        indexing_instructions = {'index' if update_mode == 'replace' else 'update': {"_index": index_name}}
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

        if update_mode == "replace":
            indexing_instructions["index"]["_id"] = doc_id
            if use_existing_tensors:
                matching_doc = [doc for doc in existing_docs["docs"] if doc["_id"] == doc_id]
                # Should only have 1 result, as only 1 id matches
                if len(matching_doc) == 1:
                    existing_doc = matching_doc[0]
                # When a request isn't sent to get matching docs, because the added docs don't
                # have IDs:
                elif len(matching_doc) == 0:
                    existing_doc = {"found": False}
        else:
            indexing_instructions["update"]["_id"] = doc_id

        # Metadata can be calculated here at the doc level.
        # Only add chunk values which are string, boolean, numeric or dictionary.
        # Dictionary keys will be store in a list.
        chunk_values_for_filtering = {}
        for key, value in copied.items():
            if not (isinstance(value, str) or isinstance(value, float)
                    or isinstance(value, bool) or isinstance(value, int)
                    or isinstance(value, list) or isinstance(value, dict)):
                continue
            if isinstance(value, dict):
                chunk_values_for_filtering[key] = list(value.keys())
            else:
                chunk_values_for_filtering[key] = value

        chunks = []

        for field in copied:

            try:
                field_content = validation.validate_field_content(
                    field_content=copied[field], is_non_tensor_field=field in non_tensor_fields
                )
            except errors.InvalidArgError as err:
                document_is_valid = False
                unsuccessful_docs.append(
                    (i, {'_id': doc_id, 'error': err.message, 'status': int(err.status_code),
                         'code': err.code})
                )
                break

            if field not in existing_fields:
                new_fields_from_doc.add((field, _infer_opensearch_data_type(copied[field])))

            # Don't process text/image fields when explicitly told not to.
            if field in non_tensor_fields:
                continue

            # chunks generated by processing this field for this doc:
            chunks_to_append = []
            # Check if content of this field changed. If no, skip all chunking and vectorisation
            if ((update_mode == 'replace') and use_existing_tensors and existing_doc["found"]
                    and (field in existing_doc["_source"]) and (existing_doc["_source"][field] == field_content)):
                # logger.info(f"Using existing vectors for doc {doc_id}, field {field}. Content remains unchanged.")
                chunks_to_append = _get_chunks_for_field(field_name=field, doc_id=doc_id, doc=existing_doc)

            # Chunk and vectorise, since content changed.
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
                                image_data, device=selected_device, method=image_method)
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
                    vector_chunks = s2_inference.vectorise(
                        model_name=index_info.model_name,
                        model_properties=_get_model_properties(index_info), content=content_chunks,
                        device=selected_device, normalize_embeddings=normalize_embeddings,
                        infer=infer_if_image)

                    end_time = timer()
                    single_vectorise_call = end_time - start_time
                    total_vectorise_time += single_vectorise_call
                    logger.debug(f"(4) TIME for single vectorise call: {(single_vectorise_call):.3f}s.")
                except (s2_inference_errors.UnknownModelError,
                        s2_inference_errors.InvalidModelPropertiesError,
                        s2_inference_errors.ModelLoadError) as model_error:
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
                        f"recevied text_chunks={len(text_chunks)} and vector_chunks={len(vector_chunks)}. "
                        f"check the preprocessing functions and try again. ")

                for text_chunk, vector_chunk in zip(text_chunks, vector_chunks):
                    # We do not put in metadata yet at this stage.
                    chunks_to_append.append({
                        utils.generate_vector_name(field): vector_chunk,
                        TensorField.field_content: text_chunk,
                        TensorField.field_name: field
                    })

            # Add chunks_to_append along with doc metadata to total chunks

            elif isinstance(field_content, dict):
                combo_chunk, combo_document_is_valid, unsuccessful_doc_to_append, combo_vectorise_time_to_add = \
                    vectorise_multimodal_combination_field(field, field_content, copied,
                                                           i, doc_id, selected_device, index_info, image_repo)

                total_vectorise_time = total_vectorise_time + combo_vectorise_time_to_add
                if combo_document_is_valid is False:
                    unsuccessful_docs.append(unsuccessful_doc_to_append)
                    break
                else:
                    chunks.append({**combo_chunk, **chunk_values_for_filtering})
                    continue

            for chunk in chunks_to_append:
                chunks.append({**chunk, **chunk_values_for_filtering})

        print(chunks)
        if document_is_valid:
            # This block happens per DOC
            new_fields = new_fields.union(new_fields_from_doc)
            if update_mode == 'replace':
                copied[TensorField.chunks] = chunks
                bulk_parent_dicts.append(indexing_instructions)
                bulk_parent_dicts.append(copied)
            else:
                to_upsert = copied.copy()
                to_upsert[TensorField.chunks] = chunks
                bulk_parent_dicts.append(indexing_instructions)
                bulk_parent_dicts.append({
                    "upsert": to_upsert,
                    "script": {
                        "lang": "painless",
                        "source": f"""
            
            // updates the doc's fields with the new content
            for (key in params.customer_dict.keySet()) {{
                ctx._source[key] = params.customer_dict[key];
            }}            
                        
            // keep track of the merged doc
            def merged_doc = [:];
            merged_doc.putAll(ctx._source);
            merged_doc.remove("{TensorField.chunks}");
            
            // remove chunks if the __field_name matches an updated field
            // All update fields should be recomputed, and it should be safe to delete these chunks             
            for (int i=ctx._source.{TensorField.chunks}.length-1; i>=0; i--) {{
                if (params.doc_fields.contains(ctx._source.{TensorField.chunks}[i].{TensorField.field_name})) {{
                   ctx._source.{TensorField.chunks}.remove(i);
                }}
                // Check if the field should have a tensor, remove if not.
                else if (params.non_tensor_fields.contains(ctx._source.{TensorField.chunks}[i].{TensorField.field_name})) {{
                    ctx._source.{TensorField.chunks}.remove(i);
                }}
            }}
            
            // update the chunks, setting fields to the new data
            for (int i=ctx._source.{TensorField.chunks}.length-1; i>=0; i--) {{
                for (key in params.customer_dict.keySet()) {{
                    ctx._source.{TensorField.chunks}[i][key] = params.customer_dict[key];
                }}
            }}
            
            // update the new chunks, adding the existing data 
            for (int i=params.new_chunks.length-1; i>=0; i--) {{
                for (key in merged_doc.keySet()) {{
                    params.new_chunks[i][key] = merged_doc[key];
                }}
            }}
            
            // appends the new chunks to the existing chunks  
            ctx._source.{TensorField.chunks}.addAll(params.new_chunks);
            
                        """,
                        "params": {
                            "doc_fields": list(copied.keys()),
                            "new_chunks": chunks,
                            "customer_dict": copied,
                            "non_tensor_fields": non_tensor_fields
                        },
                    }
                })

    end_time_3 = timer()
    total_preproc_time = end_time_3 - start_time_3
    logger.info(f"      add_documents pre-processing: took {(total_preproc_time):.3f}s total for {batch_size} docs, "
                f"for an average of {(total_preproc_time / batch_size):.3f}s per doc.")

    logger.info(f"          add_documents vectorise: took {(total_vectorise_time):.3f}s for {batch_size} docs, "
                f"for an average of {(total_vectorise_time / batch_size):.3f}s per doc.")

    if bulk_parent_dicts:
        # the HttpRequest wrapper handles error logic
        update_mapping_response = backend.add_customer_field_properties(
            config=config, index_name=index_name, customer_field_names=new_fields,
            model_properties=_get_model_properties(index_info))

        # ADD DOCS TIMER-LOGGER (5)
        start_time_5 = timer()
        index_parent_response = HttpRequests(config).post(
            path="_bulk", body=utils.dicts_to_jsonl(bulk_parent_dicts))
        end_time_5 = timer()
        total_http_time = end_time_5 - start_time_5
        total_index_time = index_parent_response["took"] * 0.001
        logger.info(
            f"      add_documents roundtrip: took {(total_http_time):.3f}s to send {batch_size} docs (roundtrip) to Marqo-os, "
            f"for an average of {(total_http_time / batch_size):.3f}s per doc.")

        logger.info(
            f"          add_documents Marqo-os index: took {(total_index_time):.3f}s for Marqo-os to index {batch_size} docs, "
            f"for an average of {(total_index_time / batch_size):.3f}s per doc.")
    else:
        index_parent_response = None

    if auto_refresh:
        refresh_response = HttpRequests(config).post(path=F"{index_name}/_refresh")

    t1 = timer()

    def translate_add_doc_response(response: Optional[dict], time_diff: float) -> dict:
        """translates OpenSearch response dict into Marqo dict"""
        item_fields_to_remove = ['_index', '_primary_term', '_seq_no', '_shards', '_version']
        result_dict = {}
        new_items = []

        if response is not None:
            copied_res = copy.deepcopy(response)

            result_dict['errors'] = copied_res['errors']
            actioned = "index" if update_mode == 'replace' else 'update'

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
        result_dict["index_name"] = index_name
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
    for doc_id in document_ids:
        # There should always be 2 results per doc.
        result_list = [doc for doc in res["docs"] if doc["_id"] == doc_id]
        if len(result_list) == 0:
            continue
        if len(result_list) not in (2, 0):
            raise errors.MarqoWebError(f"Bad request for existing documents. "
                                       f"There are {len(result_list)} results for doc id {doc_id}.")

        for result in result_list:
            if result["found"]:
                doc_in_results = True
                if result["_source"]["__chunks"] == []:
                    res_data = result
                else:
                    res_chunks = result
            else:
                doc_in_results = False
                dummy_res = result
                break

        # Put the chunks list in res_data, so it's complete
        if doc_in_results:
            res_data["_source"]["__chunks"] = res_chunks["_source"]["__chunks"]
            combined_result.append(res_data)
        else:
            # This result just says that the doc was not found
            combined_result.append(dummy_res)

    res["docs"] = combined_result

    # Returns a list of combined docs
    return res


def delete_documents(config: Config, index_name: str, doc_ids: List[str], auto_refresh):
    """Deletes documents """
    if not doc_ids:
        raise errors.InvalidDocumentIdError("doc_ids can't be empty!")

    for _id in doc_ids:
        validation.validate_id(_id)

    # TODO: change to timer()
    t0 = datetime.datetime.utcnow()
    delete_res_backend = HttpRequests(config=config).post(
        path=f"{index_name}/_delete_by_query", body={
            "query": {
                "terms": {
                    "_id": doc_ids
                }
            }
        }
    )
    if auto_refresh:
        refresh_response = HttpRequests(config).post(path=F"{index_name}/_refresh")
    t1 = datetime.datetime.utcnow()
    delete_res = {
        "index_name": index_name, "status": "succeeded",
        "type": "documentDeletion", "details": {
            "receivedDocumentIds": len(doc_ids),
            "deletedDocuments": delete_res_backend["deleted"],
        },
        "duration": utils.create_duration_string(t1 - t0),
        "startedAt": utils.format_timestamp(t0),
        "finishedAt": utils.format_timestamp(t1),
    }
    return delete_res


def refresh_index(config: Config, index_name: str):
    return HttpRequests(config).post(path=F"{index_name}/_refresh")


def search(config: Config, index_name: str, text: Union[str, dict],
           result_count: int = 3, offset: int = 0, highlights=True, return_doc_ids=True,
           search_method: Union[str, SearchMethod, None] = SearchMethod.TENSOR,
           searchable_attributes: Iterable[str] = None, verbose: int = 0, num_highlights: int = 3,
           reranker: Union[str, Dict] = None, simplified_format: bool = True, filter: str = None,
           attributes_to_retrieve: Optional[List[str]] = None,
           device=None, boost: Optional[Dict] = None,
           image_download_headers: Optional[Dict] = None) -> Dict:
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
        return_doc_ids:
        search_method:
        searchable_attributes:
        verbose:
        num_highlights: number of highlights to return for each doc
        boost: boosters to re-weight the scores of individual fields
        image_download_headers: headers for downloading images
    Returns:

    """
    # Validation for: result_count (limit) & offset
    # Validate neither is negative
    if result_count <= 0:
        raise errors.IllegalRequestedDocCount("search result limit must be greater than 0!")
    if offset < 0:
        raise errors.IllegalRequestedDocCount("search result offset cannot be less than 0!")

        # validate query
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
    validation.validate_boost(boost=boost, search_method=search_method)
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

    if search_method.upper() == SearchMethod.TENSOR:
        search_result = _vector_text_search(
            config=config, index_name=index_name, query=text, result_count=result_count, offset=offset,
            return_doc_ids=return_doc_ids, searchable_attributes=searchable_attributes, verbose=verbose,
            number_of_highlights=num_highlights, simplified_format=simplified_format,
            filter_string=filter, device=device, attributes_to_retrieve=attributes_to_retrieve, boost=boost,
            image_download_headers=image_download_headers
        )
    elif search_method.upper() == SearchMethod.LEXICAL:
        search_result = _lexical_search(
            config=config, index_name=index_name, text=text, result_count=result_count, offset=offset,
            return_doc_ids=return_doc_ids, searchable_attributes=searchable_attributes, verbose=verbose,
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
            start_rerank_time = timer()
            rerank.rerank_search_results(search_result=search_result, query=text,
                                         model_name=reranker,
                                         device=config.indexing_device if device is None else device,
                                         searchable_attributes=searchable_attributes,
                                         num_highlights=1 if simplified_format else num_highlights)
            end_rerank_time = timer()
            total_rerank_time = end_rerank_time - start_rerank_time
            logger.info(
                f"search ({search_method.lower()}) reranking using {reranker}: took {(total_rerank_time):.3f}s to rerank results.")
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
    logger.info(f"search ({search_method.lower()}) completed with total processing time: {(time_taken):.3f}s.")

    return search_result


def _lexical_search(
        config: Config, index_name: str, text: str, result_count: int = 3, offset: int = 0, return_doc_ids=True,
        searchable_attributes: Sequence[str] = None, verbose: int = 0, filter_string: str = None,
        attributes_to_retrieve: Optional[List[str]] = None, expose_facets: bool = False):
    """

    Args:
        config:
        index_name:
        text:
        result_count:
        offset:
        return_doc_ids:
        searchable_attributes:
        number_of_highlights:
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
            f"Query arg must be of type str! text arg is of type {type(text)}. "
            f"Query arg: {text}")

    # SEARCH TIMER-LOGGER (pre-processing)
    start_preprocess_time = timer()
    if searchable_attributes is not None and searchable_attributes:
        fields_to_search = searchable_attributes
    else:
        fields_to_search = index_meta_cache.get_index_info(
            config=config, index_name=index_name
        ).get_true_text_properties()

    # Validation for offset (pagination is single field)
    if len(fields_to_search) != 1 and offset > 0:
        raise errors.InvalidArgError(
            f"Pagination (offset > 0) is only supported for single field searches! Your search currently has {len(fields_to_search)} fields: {fields_to_search}")

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

    end_preprocess_time = timer()
    total_preprocess_time = end_preprocess_time - start_preprocess_time
    logger.info(f"search (lexical) pre-processing: took {(total_preprocess_time):.3f}s to process query.")

    start_search_http_time = timer()
    search_res = HttpRequests(config).get(path=f"{index_name}/_search", body=body)

    end_search_http_time = timer()
    total_search_http_time = end_search_http_time - start_search_http_time
    total_os_process_time = search_res["took"] * 0.001
    num_results = len(search_res['hits']['hits'])
    logger.info(
        f"search (lexical) roundtrip: took {(total_search_http_time):.3f}s to send search query (roundtrip) to Marqo-os and received {num_results} results.")
    logger.info(
        f"  search (lexical) Marqo-os processing time: took {(total_os_process_time):.3f}s for Marqo-os to execute the search.")

    # SEARCH TIMER-LOGGER (post-processing)
    start_postprocess_time = timer()

    res_list = []
    for doc in search_res['hits']['hits']:
        just_doc = _clean_doc(doc["_source"].copy()) if "_source" in doc else dict()
        if return_doc_ids:
            just_doc["_id"] = doc["_id"]
            just_doc["_score"] = doc["_score"]
        res_list.append({**just_doc, "_highlights": []})

    end_postprocess_time = timer()
    total_postprocess_time = end_postprocess_time - start_postprocess_time
    logger.info(
        f"search (lexical) post-processing: took {(total_postprocess_time):.3f}s to format {len(res_list)} results.")

    return {'hits': res_list}


def _vector_text_search(
        config: Config, index_name: str, query: Union[str, dict], result_count: int = 5, offset: int = 0,
        return_doc_ids=False, searchable_attributes: Iterable[str] = None, number_of_highlights=3,
        verbose=0, raise_on_searchable_attribs=False, hide_vectors=True, k=500,
        simplified_format=True, filter_string: str = None, device=None,
        attributes_to_retrieve: Optional[List[str]] = None, boost: Optional[Dict] = None,
        image_download_headers: Optional[Dict] = None):
    """
    Args:
        config:
        index_name:
        query: either a string query (which can be a URL or natural language text), or a dict of
            <query string>:<weight float> pairs.
        result_count:
        offset:
        return_doc_ids: if True adds doc _id to the docs. Otherwise just returns the docs as-is
        searchable_attributes: Iterable of field names to search. If left as None, then all will
            be searched
        number_of_highlights: if None, will return all highlights in
            descending order of relevancy. Otherwise will return this number of highlights
        verbose: if 0 - nothing is printed. if 1 - data is printed without vectors, if 2 - full
            objects are printed out
        attributes_to_retrieve: if set, only returns these fields
        image_download_headers: headers for downloading images
    Returns:

    Note:
        - uses multisearch, which returns k results in each attribute. Not that much of a concern unless you have a
        ridiculous number of attributes
        - Should not be directly called by client - the search() method should
        be called. The search() method adds syncing

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
    # SEARCH TIMER-LOGGER (pre-processing)
    start_preprocess_time = timer()
    try:
        index_info = get_index_info(config=config, index_name=index_name)
    except KeyError as e:
        raise errors.IndexNotFoundError(message="Tried to search a non-existent index: {}".format(index_name))
    selected_device = config.indexing_device if device is None else device

    # query, weight pairs, if query is a dict:
    ordered_queries = None

    if isinstance(query, str):
        to_be_vectorised = [query, ]
    else:  # is dict:
        ordered_queries = list(query.items())
        if index_info.index_settings[NsField.index_defaults][NsField.treat_urls_and_pointers_as_images]:
            text_queries = [k for k, _ in ordered_queries if _is_image(k)]
            image_queries = [k for k, _ in ordered_queries if not _is_image(k)]
            to_be_vectorised = [batch for batch in [text_queries, image_queries] if batch]
        else:
            to_be_vectorised = [[k for k, _ in ordered_queries], ]
    try:
        vectorised_text = functools.reduce(lambda x, y: x + y,
                                           [s2_inference.vectorise(
                                               model_name=index_info.model_name,
                                               model_properties=_get_model_properties(index_info),
                                               content=batch, device=selected_device,
                                               normalize_embeddings=index_info.index_settings['index_defaults'][
                                                   'normalize_embeddings'],
                                               image_download_headers=image_download_headers
                                           )
                                               for batch in to_be_vectorised]
                                           )
        if ordered_queries:
            # multiple queries. We have to weight and combine them:
            weighted_vectors = [np.asarray(vec) * weight for vec, weight in
                                zip(vectorised_text, [w for _, w in ordered_queries])]
            vectorised_text = np.mean(weighted_vectors, axis=0)
            if index_info.index_settings['index_defaults']['normalize_embeddings']:
                norm = np.linalg.norm(vectorised_text, axis=-1, keepdims=True)
                if norm > 0:
                    vectorised_text /= np.linalg.norm(vectorised_text, axis=-1, keepdims=True)
            vectorised_text = list(vectorised_text)
        else:
            vectorised_text = vectorised_text[0]
    except (s2_inference_errors.UnknownModelError,
            s2_inference_errors.InvalidModelPropertiesError,
            s2_inference_errors.ModelLoadError) as model_error:
        raise errors.BadRequestError(
            message=f'Problem vectorising query. Reason: {str(model_error)}',
            # link="https://marqo.pages.dev/latest/Models-Reference/dense_retrieval/"
        )
    except s2_inference_errors.S2InferenceError as s2_error:
        raise errors.BadRequestError(
            message=f"Problem vectorising query. Reason: {str(s2_error)}"
        )
    body = []

    if searchable_attributes is None:
        vector_properties_to_search = index_info.get_vector_properties().keys()
    else:
        if raise_on_searchable_attribs:
            vector_properties_to_search = validation.validate_searchable_vector_props(
                existing_vector_properties=index_info.get_vector_properties().keys(),
                subset_vector_properties=searchable_attributes
            )
        else:
            searchable_attributes_as_vectors = {utils.generate_vector_name(field_name=attribute)
                                                for attribute in searchable_attributes}
            # discard searchable attributes that aren't found in the cache:
            vector_properties_to_search = searchable_attributes_as_vectors.intersection(
                index_info.get_vector_properties().keys())

    # Validation for offset (pagination is single field)
    if len(vector_properties_to_search) != 1 and offset > 0:
        human_readable_vector_properties = [v.replace(TensorField.vector_prefix, "") for v in
                                            list(vector_properties_to_search)]
        raise errors.InvalidArgError(
            f"Pagination (offset > 0) is only supported for single field searches! Your search currently has {len(vector_properties_to_search)} vectorisable fields: {human_readable_vector_properties}")

    if filter_string is not None:
        contextualised_filter = utils.contextualise_filter(
            filter_string=filter_string,
            simple_properties=index_info.get_text_properties())
    else:
        contextualised_filter = ''

    for vector_field in vector_properties_to_search:
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

        field_names = list(index_info.get_text_properties().keys())
        if attributes_to_retrieve is not None:
            search_query["_source"] = {"include": attributes_to_retrieve} if len(attributes_to_retrieve) > 0 else False

        if filter_string is not None:
            search_query["query"]["nested"]["query"]["knn"][f"{TensorField.chunks}.{vector_field}"][
                "filter"] = {
                "query_string": {"query": f"{contextualised_filter}"}
            }
        body += [{"index": index_name}, search_query]

    if verbose:
        print("vector search body:")
        if verbose == 1:
            readable_body = copy.deepcopy(body)
            for i, q in enumerate(readable_body):
                if "index" in q:
                    continue
                for vec in list(q["query"]["nested"]["query"]["knn"].keys()):
                    readable_body[i]["query"]["nested"]["query"]["knn"][vec]["vector"] = \
                        readable_body[i]["query"]["nested"]["query"]["knn"][vec]["vector"][:5]
            pprint.pprint(readable_body)
        if verbose == 2:
            pprint.pprint(body, compact=True)

    if not body:
        # empty body means that there are no vector fields associated with the index.
        # This probably means the index is emtpy
        return {"hits": []}

    end_preprocess_time = timer()
    total_preprocess_time = end_preprocess_time - start_preprocess_time
    logger.info(f"search (tensor) pre-processing: took {(total_preprocess_time):.3f}s to vectorize and process query.")

    # SEARCH TIMER-LOGGER (roundtrip)
    start_search_http_time = timer()
    response = HttpRequests(config).get(path=F"{index_name}/_msearch", body=utils.dicts_to_jsonl(body))

    end_search_http_time = timer()
    total_search_http_time = end_search_http_time - start_search_http_time
    total_os_process_time = response["took"] * 0.001
    num_responses = len(response["responses"])
    logger.info(
        f"search (tensor) roundtrip: took {(total_search_http_time):.3f}s to send {num_responses} search queries (roundtrip) to Marqo-os.")

    try:
        responses = [r['hits']['hits'] for r in response["responses"]]

        # SEARCH TIMER-LOGGER (Log number of results and time for each search in multisearch)
        for i in range(len(vector_properties_to_search)):
            indiv_responses = response["responses"][i]['hits']['hits']
            indiv_query_time = response["responses"][i]["took"] * 0.001
            logger.info(
                f"  search (tensor) Marqo-os processing time (search field = {list(vector_properties_to_search)[i]}): took {(indiv_query_time):.3f}s and received {len(indiv_responses)} hits.")

    except KeyError as e:
        # KeyError indicates we have received a non-successful result
        try:
            if "index.max_result_window" in response["responses"][0]["error"]["root_cause"][0]["reason"]:
                raise errors.IllegalRequestedDocCount(
                    "Marqo-OS rejected the response due to too many requested results. "
                    "Try reducing the query's limit parameter") from e
            elif 'parse_exception' in response["responses"][0]["error"]["root_cause"][0]["reason"]:
                raise errors.InvalidArgError("Syntax error, could not parse filter string") from e
            elif (contextualised_filter
                  and contextualised_filter in response["responses"][0]["error"]["root_cause"][0]["reason"]):
                raise errors.InvalidArgError("Syntax error, could not parse filter string") from e
            raise errors.BackendCommunicationError(f"Error communicating with Marqo-OS backend:\n{response}")
        except (KeyError, IndexError) as e2:
            raise e

    logger.info(
        f"  search (tensor) Marqo-os processing time: took {(total_os_process_time):.3f}s for Marqo-os to execute the search.")

    # SEARCH TIMER-LOGGER (post-processing)
    start_postprocess_time = timer()
    gathered_docs = dict()

    if verbose:
        print("search responses:")
        pprint.pprint(responses)
    for i, query_res in enumerate(responses):
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

    def boost_score(docs: dict, boosters: dict) -> dict:
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
                        chunk['_score'] = chunk['_score'] * booster[0] + booster[1]
                    else:
                        chunk['_score'] = chunk['_score'] * booster[0]
                    boosted_fields.add(field_name)
        return to_be_boosted

    # SORT THE DOCS HERE
    def sort_chunks(docs: dict) -> dict:
        to_be_sorted = docs.copy()
        for doc_id in list(to_be_sorted.keys()):
            to_be_sorted[doc_id]["chunks"] = sorted(
                to_be_sorted[doc_id]["chunks"], key=lambda x: x["_score"], reverse=True)
        return to_be_sorted

    if boost is not None:
        docs_chunk_boosted = boost_score(gathered_docs, boost)
        docs_chunks_sorted = sort_chunks(docs_chunk_boosted)
    else:
        docs_chunks_sorted = sort_chunks(gathered_docs)

    def sort_docs(docs: dict) -> List[dict]:
        as_list = list(docs.values())
        return sorted(as_list, key=lambda x: x["chunks"][0]["_score"], reverse=True)

    completely_sorted = sort_docs(docs_chunks_sorted)

    if verbose:
        print("Chunk vector search, sorted result:")
        if verbose == 1:
            pprint.pprint(utils.truncate_dict_vectors(completely_sorted))
        elif verbose == 2:
            pprint.pprint(completely_sorted)

    # format output:
    def format_ordered_docs_preserving(ordered_docs_w_chunks: List[dict], num_highlights: Optional[int]) -> dict:
        """Formats docs so that it preserves the original document, unless doc_ids are returned
        Args:
            ordered_docs_w_chunks:
            num_highlights: number of highlights to return.
        Returns:
        """
        return {'hits': [dict([
            ('doc', _clean_doc(doc['doc']["_source"], doc_id=doc['_id'] if return_doc_ids else None)),
            ('highlights', [{
                the_chunk["_source"][TensorField.field_name]: the_chunk["_source"][TensorField.field_content]
            } for the_chunk in doc['chunks']][:num_highlights])
        ]) for doc in ordered_docs_w_chunks][:result_count]}

    # format output:
    def format_ordered_docs_simple(ordered_docs_w_chunks: List[dict]) -> dict:
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

    if simplified_format:
        res = format_ordered_docs_simple(ordered_docs_w_chunks=completely_sorted)
    else:
        res = format_ordered_docs_preserving(ordered_docs_w_chunks=completely_sorted,
                                             num_highlights=number_of_highlights)

    end_postprocess_time = timer()
    total_postprocess_time = end_postprocess_time - start_postprocess_time
    logger.info(
        f"search (tensor) post-processing: took {(total_postprocess_time):.3f}s to sort and format {len(completely_sorted)} results from Marqo-os.")
    return res


def check_health(config: Config):
    TIMEOUT = 3
    statuses = {
        "green": 0,
        "yellow": 1,
        "red": 2
    }

    marqo_status = "green"
    marqo_os_health_check = None
    try:
        timeout_config = copy.deepcopy(config)
        timeout_config.timeout = TIMEOUT
        marqo_os_health_check = HttpRequests(timeout_config).get(
            path="_cluster/health"
        )
    except errors.BackendCommunicationError:
        marqo_os_status = "red"

    if marqo_os_health_check is not None:
        if "status" in marqo_os_health_check:
            marqo_os_status = marqo_os_health_check['status']
        else:
            marqo_os_status = "red"
    else:
        marqo_os_status = "red"

    marqo_status = marqo_status if statuses[marqo_status] >= statuses[marqo_os_status] else marqo_os_status

    return {
        "status": marqo_status,
        "backend": {
            "status": marqo_os_status
        }
    }


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


def _get_model_properties(index_info):
    index_defaults = index_info.get_index_settings()["index_defaults"]
    try:
        model_properties = index_defaults[NsField.model_properties]
    except KeyError:
        try:
            model_properties = s2_inference.get_model_properties_from_registry(index_info.model_name)
        except s2_inference_errors.UnknownModelError:
            raise s2_inference_errors.UnknownModelError(
                f"Could not find model properties for model={index_info.model_name}. "
                f"Please check that the model name is correct. "
                f"Please provide model_properties if the model is a custom model and is not supported by default")

    return model_properties


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


def vectorise_multimodal_combination_field(field: str, field_content: Dict[str, dict], doc: dict, doc_index: int,
                                            doc_id:str, selected_device:str, index_info, image_repo):
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
        selected_device: device from main body
        index_info: index_info from main body
    Returns:
    combo_chunk_to_append: the combo_chunk to be appended to the main body
    combo_document_is_valid:  if the document is a valid
    unsuccessful_docs: appended unsucessful_docs
    combo_total_vectorise_time: the vectorise time spent in combo field
    '''
    # field_conent = {"tensor_field_one" : {"weight":0.5, "parameter": "test-paramater-1"},
    #                 "tensor_field_two" : {"weight": 0.5, parameter": "test-parameter-2"}},
    combo_document_is_valid = True
    field_vector_weight = []
    combo_vectorise_time_to_add = 0
    combo_chunk = {}
    unsuccessful_doc_to_append = tuple()

    # Copy the important mutable objects from main body for safety purpose
    field_content_copy = copy.deepcopy(field_content)
    doc_copy = copy.deepcopy(doc)

    # 4 list to store content_chunks, text_chunks to image and text.
    # It can also be two list of tuples, but i don't think they can be dictioanry.
    # I choose to use 4 lists as they are easy to handle.



    text_content_chunks = []
    text_text_chunks = []

    image_content_chunks = []
    image_text_chunks = []

    for sub_content in list(field_content_copy):
        if isinstance(sub_content, str) and not _is_image(sub_content):
            text_content_chunks.append(sub_content)
            text_text_chunks.append(sub_content)
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

                image_content_chunks.append(image_data)
                image_text_chunks.append(sub_content)

            except s2_inference_errors.S2InferenceError as e:
                combo_document_is_valid = False
                unsuccessful_doc_to_append = \
                    (doc_index, {'_id': doc_id, 'error': e.message,
                                 'status': int(errors.InvalidArgError.status_code),
                                 'code': errors.InvalidArgError.code})

                return combo_chunk, combo_document_is_valid, unsuccessful_doc_to_append, combo_vectorise_time_to_add

    normalize_embeddings = index_info.index_settings[NsField.index_defaults][
        NsField.normalize_embeddings]
    infer_if_image = index_info.index_settings[NsField.index_defaults][
        NsField.treat_urls_and_pointers_as_images]


    try:
        start_time = timer()

        text_vector_chunks = s2_inference.vectorise(
            model_name=index_info.model_name,
            model_properties=_get_model_properties(index_info), content=text_content_chunks,
            device=selected_device, normalize_embeddings=normalize_embeddings,
            # it should always be false for multimodal-combination
            infer=infer_if_image)

        image_vector_chunks = s2_inference.vectorise(
            model_name=index_info.model_name,
            model_properties=_get_model_properties(index_info), content=image_content_chunks,
            device=selected_device, normalize_embeddings=normalize_embeddings,
            # it should always be false for multimodal-combination
            infer=infer_if_image)

        end_time = timer()
        single_vectorise_call = end_time - start_time
        combo_vectorise_time_to_add += single_vectorise_call
        logger.debug(f"(4) TIME for single vectorise call: {(single_vectorise_call):.3f}s.")
    except (s2_inference_errors.UnknownModelError,
            s2_inference_errors.InvalidModelPropertiesError,
            s2_inference_errors.ModelLoadError) as model_error:
        raise errors.BadRequestError(
            message=f'Problem vectorising query. Reason: {str(model_error)}',
            link="https://marqo.pages.dev/latest/Models-Reference/dense_retrieval/"
        )
    except s2_inference_errors.S2InferenceError:
        combo_document_is_valid = False
        image_err = errors.InvalidArgError(message=f'Could not process given image: {field_content_copy}')
        unsuccessful_doc_to_append = \
            (doc_index, {'_id': doc_id, 'error': image_err.message, 'status': int(image_err.status_code),
                 'code': image_err.code})

        return combo_chunk, combo_document_is_valid, unsuccessful_doc_to_append, combo_vectorise_time_to_add

    content_list = text_text_chunks + image_text_chunks
    vectors_list = text_vector_chunks + image_vector_chunks

    vector_chunk = np.squeeze(np.sum([np.array(vector) * field_content_copy[content]["weight"] for content, vector in zip(content_list, vectors_list)], axis=0))

    if normalize_embeddings is True:
        vector_chunk = vector_chunk / np.linalg.norm(vector_chunk)

    vector_chunk = vector_chunk.tolist()

    combo_chunk = dict({
        utils.generate_vector_name(field): vector_chunk,
        TensorField.field_content: list(doc_copy[field].keys()), # replace text_chunks
        TensorField.field_name: field,
    })
    print(combo_chunk)
    return combo_chunk, combo_document_is_valid, unsuccessful_doc_to_append, combo_vectorise_time_to_add









    # for sub_content, sub_content_para in field_content_copy.items():
    #     if isinstance(sub_content, str):
    #         if isinstance(sub_content, str) and not _is_image(sub_content):
    #             text_chunks = sub_content
    #             content_chunks = text_chunks
    #             text_content_list.append((text_chunks, content_chunks))
    #         else:
    #             try:
    #                 if isinstance(sub_content, str) and index_info.index_settings[NsField.index_defaults][
    #                     NsField.treat_urls_and_pointers_as_images]:
    #                     if not isinstance(image_repo[sub_content], Exception):
    #                         image_data = image_repo[sub_content]
    #                     else:
    #                         raise s2_inference_errors.S2InferenceError(
    #                             f"Could not find image found at `{sub_content}`. \n"
    #                             f"Reason: {str(image_repo[sub_content])}"
    #                         )
    #                 else:
    #                     image_data = sub_content
    #
    #                 content_chunks, text_chunks = [image_data], [sub_content]
    #                 image_content_list.append((text_chunks, content_chunks))
    #
    #             except s2_inference_errors.S2InferenceError as e:
    #                 combo_document_is_valid = False
    #                 unsuccessful_doc_to_append =\
    #                     (doc_index, {'_id': doc_id, 'error': e.message,
    #                          'status': int(errors.InvalidArgError.status_code),
    #                          'code': errors.InvalidArgError.code})
    #
    #                 return combo_chunk, combo_document_is_valid, unsuccessful_doc_to_append, combo_vectorise_time_to_add
    #
    # normalize_embeddings = index_info.index_settings[NsField.index_defaults][
    #     NsField.normalize_embeddings]
    # infer_if_image = index_info.index_settings[NsField.index_defaults][
    #     NsField.treat_urls_and_pointers_as_images]
    #
    # try:
    #     if infer_if_image is False:
    #         raise errors.InvalidArgError(
    #             f"The setting `treat_urls_and_pointers_as_images` is `{infer_if_image}` for a multimodal_tensor_combination field."
    #             f"This is not supported. Please change the setting `treat_urls_and_pointers_as_images` as `True` for multimodal_tensor_combination fields. "
    #         )
    # except errors.InvalidArgError as e:
    #     combo_document_is_valid = False
    #     unsuccessful_doc_to_append = \
    #         (doc_index, {'_id': doc_id, 'error': e.message,
    #              'status': int(errors.InvalidArgError.status_code),
    #              'code': errors.InvalidArgError.code})
    #
    #     return combo_chunk, combo_document_is_valid, unsuccessful_doc_to_append, combo_vectorise_time_to_add
    # try:
    #     start_time = timer()
    #     vector_chunks = s2_inference.vectorise(
    #         model_name=index_info.model_name,
    #         model_properties=_get_model_properties(index_info), content=content_chunks,
    #         device=selected_device, normalize_embeddings=normalize_embeddings,
    #         # it should always be false for multimodal-combination
    #         infer=infer_if_image)
    #
    #     end_time = timer()
    #     single_vectorise_call = end_time - start_time
    #     combo_vectorise_time_to_add += single_vectorise_call
    #     logger.debug(f"(4) TIME for single vectorise call: {(single_vectorise_call):.3f}s.")
    # except (s2_inference_errors.UnknownModelError,
    #         s2_inference_errors.InvalidModelPropertiesError,
    #         s2_inference_errors.ModelLoadError) as model_error:
    #     raise errors.BadRequestError(
    #         message=f'Problem vectorising query. Reason: {str(model_error)}',
    #         link="https://marqo.pages.dev/latest/Models-Reference/dense_retrieval/"
    #     )
    # except s2_inference_errors.S2InferenceError:
    #     combo_document_is_valid = False
    #     image_err = errors.InvalidArgError(message=f'Could not process given image: {field_content_copy}')
    #     unsuccessful_doc_to_append = \
    #         (doc_index, {'_id': doc_id, 'error': image_err.message, 'status': int(image_err.status_code),
    #              'code': image_err.code})
    #
    #     return combo_chunk, combo_document_is_valid, unsuccessful_doc_to_append, combo_vectorise_time_to_add
    #
    # field_vector_weight.append((sub_content_para["weight"], vector_chunks))
    #
    # vector_chunk = np.squeeze(
    #     np.sum([np.array(vector) * weight for weight, vector in field_vector_weight], axis=0))
    #
    # # Normalize the vector if normalize_embeddings = True
    # if normalize_embeddings is True:
    #     vector_chunk = vector_chunk / np.linalg.norm(vector_chunk)
    #
    # vector_chunk = vector_chunk.tolist()
    #
    # combo_chunk = dict({
    #     utils.generate_vector_name(field): vector_chunk,
    #     TensorField.field_content: list(doc_copy[field].keys()), # replace text_chunks
    #     TensorField.field_name: field,
    # })
    # return combo_chunk, combo_document_is_valid, unsuccessful_doc_to_append, combo_vectorise_time_to_add
