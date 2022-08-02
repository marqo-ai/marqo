"""neural search logic. In the future this will be accessible to the client via an API

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
import datetime
import functools
import json
import pprint
import typing
import uuid
import asyncio
from typing import List, Optional, Union, Callable, Iterable, Sequence, Dict
import requests
from marqo.neural_search.enums import MediaType, MlModel, NeuralField, SearchMethod
from marqo.neural_search.enums import NeuralSettingsField as NsField
from marqo.neural_search import utils, backend, validation, configs
from marqo.processing import text as text_processor
from marqo.s2_inference import s2_inference
from marqo.neural_search.index_meta_cache import get_cache,get_index_info
from marqo.neural_search import index_meta_cache
from marqo.neural_search.models.index_info import IndexInfo
from marqo.neural_search import constants
# We depend on _httprequests.py for now, but this may be replaced in the future, as
# _httprequests.py is designed for the client
from marqo._httprequests import HttpRequests
from marqo.config import Config
from marqo.errors import MarqoError, MarqoApiError
import threading


def create_vector_index(
        config: Config, index_name: str, media_type: Union[str, MediaType] = MediaType.default,
        refresh_interval: str = "1s", number_of_shards=5, neural_settings = None):
    """
    Args:
        media_type: 'text'|'image'
    """

    vector_index_settings = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
                "refresh_interval":  refresh_interval
            },
            "number_of_shards": number_of_shards
        },
        "mappings": {
            "_meta": {
                "media_type": media_type,
            },
            "properties": {
                NeuralField.chunks: {
                    "type": "nested",
                    "properties": {
                        NeuralField.field_name: {
                            "type": "keyword"
                        },
                        NeuralField.field_content: {
                            "type": "text"
                        },
                    }
                }
            }
        }
    }

    if neural_settings is not None:
        the_neural_settings = _autofill_neural_settings(neural_settings=neural_settings)
    else:
        the_neural_settings = configs.get_default_neural_index_settings()

    model_name = the_neural_settings[NsField.index_defaults][NsField.model]
    vector_index_settings["mappings"]["_meta"][NsField.neural_settings] = the_neural_settings
    vector_index_settings["mappings"]["_meta"]["model"] = model_name

    response = HttpRequests(config).put(path=index_name, body=vector_index_settings)

    get_cache()[index_name] = IndexInfo(
        model_name=model_name, properties=vector_index_settings["mappings"]["properties"].copy(),
        neural_settings=the_neural_settings
    )
    return response


def _autofill_neural_settings(neural_settings: dict):
    """A half-complete neural settings will be auto filled"""

    # TODO: validated conflicting settings
    # treat_urls_and_pointers_as_images

    copied_settings = neural_settings.copy()
    default_settings = configs.get_default_neural_index_settings()

    if NsField.index_defaults not in copied_settings:
        copied_settings[NsField.index_defaults] = default_settings[NsField.index_defaults]

    if NsField.treat_urls_and_pointers_as_images in copied_settings[NsField.index_defaults] and \
            copied_settings[NsField.index_defaults][NsField.treat_urls_and_pointers_as_images] is True\
            and copied_settings[NsField.index_defaults][NsField.model] is None:
        copied_settings[NsField.index_defaults][NsField.model] = MlModel.clip

    for key in list(default_settings[NsField.index_defaults]):
        if key not in copied_settings[NsField.index_defaults] or \
                copied_settings[NsField.index_defaults][key] is None:
            copied_settings[NsField.index_defaults][key] = default_settings[NsField.index_defaults][key]

    for key in list(default_settings[NsField.index_defaults][NsField.text_preprocessing]):
        if key not in copied_settings[NsField.index_defaults][NsField.text_preprocessing] or \
                copied_settings[NsField.index_defaults][NsField.text_preprocessing][key] is None:
            copied_settings[NsField.index_defaults][NsField.text_preprocessing][key] \
                = default_settings[NsField.index_defaults][NsField.text_preprocessing][key]

    return copied_settings


def get_stats(config: Config, index_name: str):
    doc_count = HttpRequests(config).post(path=F"{index_name}/_count")["count"]
    return {
        "numberOfDocuments": doc_count
    }


def add_documents(config: Config, index_name: str, docs: List[dict], auto_refresh):
    """
    FIXME:
        - Atomicity issue: we index parent doc OK, but run into error indexing chunks.
            This means we have a parent doc unsearchable via vector search!
        - Designed for only a 1-shard index!
        - Appropriate error handler if chunks can't be added to index
    """

    bulk_parent_dicts = []

    try:
        index_info = backend.get_index_info(config=config, index_name=index_name)
    except MarqoApiError as s:
        if s.status_code == 404 and "index_not_found_exception" in str(s):
            create_vector_index(config=config, index_name=index_name)
            index_info = backend.get_index_info(config=config, index_name=index_name)
        else:
            raise s

    existing_fields = set(index_info.properties.keys())
    new_fields = set()
    doc_ids_to_update = []

    for doc in docs:

        indexing_instructions = {"index": {"_index": index_name}}
        copied = doc.copy()

        validation.validate_doc(doc)
        [validation.validate_field_name(field) for field in copied]

        if "_id" in doc:
            doc_id = doc["_id"]
            if not isinstance(doc_id, str):
                raise MarqoError("Document _id must be a string type! "
                                    f"Received _id {doc_id} of type {type(doc_id)}")
            del copied["_id"]
            doc_ids_to_update.append(doc_id)
        else:
            doc_id = str(uuid.uuid4())

        indexing_instructions["index"]["_id"] = doc_id

        chunks = []

        for field in copied:

            if field not in existing_fields:
                new_fields.add(field)

            field_content = copied[field]

            if isinstance(field_content, str):
                text_chunks = text_processor.split_text(field_content)
                vector_chunks = s2_inference.vectorise(index_info.model_name, text_chunks, 
                                                    config.indexing_device, index_info.neural_settings['index_defaults']['normalize_embeddings'],
                                                    infer=index_info.neural_settings[NsField.index_defaults][NsField.treat_urls_and_pointers_as_images])
                assert len(vector_chunks) == len(text_chunks)
                for text_chunk, vector_chunk in zip(text_chunks, vector_chunks):
                    chunk_id = str(uuid.uuid4())
                    chunks.append({
                        utils.generate_vector_name(field): vector_chunk,
                        NeuralField.field_content: text_chunk,
                        NeuralField.field_name: field,
                    })
        copied[NeuralField.chunks] = chunks
        bulk_parent_dicts.append(indexing_instructions)
        bulk_parent_dicts.append(copied)

    # the HttpRequest wrapper handles error logic
    update_mapping_response = backend.add_customer_field_properties(
        config=config, index_name=index_name, customer_field_names=new_fields,
        model_properties=s2_inference.get_model_properties(model_name=index_info.model_name))

    index_parent_response = HttpRequests(config).post(
        path="_bulk", body=utils.dicts_to_jsonl(bulk_parent_dicts))

    if auto_refresh:
        refresh_response = HttpRequests(config).post(path=F"{index_name}/_refresh")

    def translate_add_doc_response(response: dict) -> dict:
        """translates OpenSearch response dict into Marqo dict"""
        copied_res = copy.deepcopy(response)
        took = copied_res["took"]
        del copied_res["took"]
        new_items = []
        item_fields_to_remove = ['_index', '_primary_term', '_seq_no', '_shards', '_version']
        for item in copied_res["items"]:
            for to_remove in item_fields_to_remove:
                del item["index"][to_remove]
            new_items.append(item["index"])
        copied_res["processingTimeMs"] = took
        copied_res["index_name"] = index_name
        copied_res["items"] = new_items
        return copied_res

    return translate_add_doc_response(index_parent_response)


def get_document_by_id(config: Config, index_name:str, document_id: str):
    """returns document by its ID"""
    if not isinstance(document_id, str):
        raise MarqoError("Document ID must be a str")
    if not document_id:
        raise MarqoError("Document ID must can't be empty")
    res = HttpRequests(config).get(
        f'{index_name}/_doc/{document_id}'
    )
    if "_source" in res:
        return _clean_doc(res["_source"], doc_id=document_id)
    else:
        return res


def delete_documents(config: Config, index_name: str, doc_ids: List[str], auto_refresh):
    """Deletes documents """
    if not doc_ids:
        raise MarqoError("doc_ids can't be empty!")
    delete_parents_res = HttpRequests(config=config).post(
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
    return delete_parents_res


def search(config: Config, index_name: str, text: str, result_count: int = 3, highlights=True, return_doc_ids=False,
           search_method: Union[str, SearchMethod, None] = SearchMethod.NEURAL,
           searchable_attributes: Iterable[str] = None, verbose=0, num_highlights=3) -> Dict:
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
        return_doc_ids:
        search_method:
        searchable_attributes:
        verbose:
        num_highlights: number of highlights to return for each doc

    Returns:

    """
    MAX_RESULT_COUNT = 500

    if result_count > MAX_RESULT_COUNT or result_count < 0:
        raise MarqoError("result count must be between 0 and 500!")

    t0 = datetime.datetime.now()

    if searchable_attributes is not None:
        [validation.validate_field_name(attribute) for attribute in searchable_attributes]

    if verbose:
        print(f"determined_search_method: {search_method}, text query: {text}")
    # if we can't see the index name in cache, we request it and wait for the info
    if index_name not in index_meta_cache.get_cache():
        backend.get_index_info(config=config, index_name=index_name)

    # update cache in the background
    cache_update_thread = threading.Thread(
        target=index_meta_cache.refresh_index,
        args=(config, index_name))
    cache_update_thread.start()

    if search_method.upper() == SearchMethod.NEURAL:
        search_result = _vector_text_search(
            config=config, index_name=index_name, text=text, result_count=result_count,
            return_doc_ids=return_doc_ids, searchable_attributes=searchable_attributes,
            number_of_highlights=num_highlights
        )
    elif search_method.upper() == SearchMethod.LEXICAL:
        search_result = _lexical_search(
            config=config, index_name=index_name, text=text, result_count=result_count,
            return_doc_ids=return_doc_ids, searchable_attributes=searchable_attributes,
        )
    else:
        raise MarqoError(f"Search called with unknown search method: {search_method}")
    time_taken = datetime.datetime.now() - t0
    search_result["processingTimeMs"] = round(time_taken.total_seconds() * 1000)
    search_result["query"] = text
    search_result["limit"] = result_count

    if not highlights:
        for hit in search_result["hits"]:
            del hit["_highlights"]

    return search_result


def _lexical_search(
        config: Config, index_name: str, text: str, result_count: int = 3, return_doc_ids=False,
        searchable_attributes: Sequence[str] = None, raise_for_searchable_attributes=False):
    """

    Args:
        config:
        index_name:
        text:
        result_count:
        return_doc_ids:
        searchable_attributes:
        number_of_highlights:
        verbose:
        raise_for_searchable_attributes: if True, check searchable attributes and raises
            and error if identified. This is appropriate only if we assume that the cache
            remains in sync with the index. If that syncing assumption is lost, then this
            behaviour may be unexpected.

    Returns:

    Notes:
        Should not be directly called by client - the search() method should
        be called. The search() method adds syncing
    TODO:
        - Test raise_for_searchable_attribute=False
    """
    if not isinstance(text, str):
        raise MarqoError(f"text arg must be of type str! text arg is of type {type(text)}. "
                            f"text arg: {text}")

    if searchable_attributes is not None:
        cached_fields_to_search = [field for field in index_meta_cache.get_index_info(
                            config=config, index_name=index_name).get_text_properties()
                            if field in searchable_attributes]
        if raise_for_searchable_attributes and len(cached_fields_to_search) < len(searchable_attributes):
            raise MarqoError(
                "Detected unknown searchable_attributes:\n "
                f"Known attributes: {index_meta_cache.get_index_info(config=config, index_name=index_name).get_text_properties().keys()},\n"
                f"Searchable attributes: {searchable_attributes}")
        else:
            fields_to_search = searchable_attributes
    else:
        fields_to_search = index_meta_cache.get_index_info(
            config=config, index_name=index_name
        ).get_text_properties()

    search_res = HttpRequests(config).get(path=f"{index_name}/_search", body={
        "query": {
            "bool": {
                "should": [
                    {"match": {field: text}}
                    for field in fields_to_search
                ],
                "must_not": [
                    {"exists": {"field": NeuralField.field_name}}
                ]
            }
        },
        "size": result_count,
    })
    res_list = []
    for doc in search_res['hits']['hits']:
        just_doc = _clean_doc(doc["_source"].copy())
        if return_doc_ids:
            just_doc["_id"] = doc["_id"]
            just_doc["_score"] = doc["_score"]
        res_list.append({**just_doc, "_highlights": []})
    return {'hits': res_list[:result_count]}


def _vector_text_search(
        config: Config, index_name: str, text: str, result_count: int = 5, return_doc_ids=False,
        searchable_attributes: Iterable[str] = None, number_of_highlights=3,
        verbose=0, raise_on_searchable_attribs=False, hide_vectors=True, k=500,
        simplified_format=True
):
    """
    Args:
        config:
        index_name:
        text:
        result_count:
        return_doc_ids: if True adds doc _id to the docs. Otherwise just returns the docs as-is
        searchable_attributes: Iterable of field names to search. If left as None, then all will
            be searched
        number_of_highlights: if None, will return all highlights in
            descending order of relevancy. Otherwise will return this number of highlights
        verbose: if 0 - nothing is printed. if 1 - data is printed without vectors, if 2 - full
            objects are printed out
        hide_vectors: if True, vectors won't be returned from OpenSearch. This reduces the size
            of data transfers
    Returns:

    Note:
        - looks for k results in each attribute. Not that much of a concern unless you have a
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
    if result_count < 0 or result_count > constants.MAX_VECTOR_SEARCH_RESULT_COUNT:
        raise ValueError("neural_search: vector_text_search: illegal result_count: {}".format(result_count))

    try:
        index_info = get_index_info(config=config, index_name=index_name)
    except KeyError as e:
        raise MarqoError(message="Tried to search a non-existent index: {}".format(index_name))
    vectorised_text = s2_inference.vectorise(
        model_name=index_info.model_name, content=text_processor.split_text(text)[0], 
        device=config.search_device,
        normalize_embeddings=index_info.neural_settings['index_defaults']['normalize_embeddings'])[0]

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

    for vector_field in vector_properties_to_search:
        search_query = {
            "size": result_count,
            "query": {
                "nested": {
                    "path": NeuralField.chunks,
                    "inner_hits": {
                        "_source": {
                            "exclude": ["*__vector*"]
                        }
                    },
                    "query": {
                        "knn": {
                            f"{NeuralField.chunks}.{vector_field}": {
                                "vector": vectorised_text,
                                "k": k
                            }
                        }
                    },
                    "score_mode": "max",
                }
            }
        }
        if hide_vectors:
            search_query["_source"] = {
                "exclude": ["*__vector*"]
            }
            search_query["query"]["nested"]["inner_hits"]["_source"] = {
                "exclude": ["*__vector*"]
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
                    readable_body[i]["query"]["nested"]["query"]["knn"][vec]["vector"] = readable_body[i]["query"]["nested"]["query"]["knn"][vec]["vector"][:5]
            pprint.pprint(readable_body)
        if verbose == 2:
            pprint.pprint(body, compact=True)

    if not body:
        # empty body means that there are no vector fields associated with the index.
        # This probably means the index is emtpy
        return {"hits": []}
    response = HttpRequests(config).get(path=F"{index_name}/_msearch", body=utils.dicts_to_jsonl(body))

    responses = [r['hits']['hits'] for r in response["responses"]]
    gathered_docs = dict()

    if verbose:
        print("search responses:")
        pprint.pprint(responses)
    for i, query_res in enumerate(responses):
        for doc in query_res:
            doc_chunks = doc["inner_hits"][NeuralField.chunks]["hits"]["hits"]
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

    # SORT THE DOCS HERE

    def sort_chunks(docs: dict) -> dict:
        to_be_sorted = docs.copy()
        for doc_id in list(to_be_sorted.keys()):
            to_be_sorted[doc_id]["chunks"] = sorted(
                to_be_sorted[doc_id]["chunks"], key=lambda x: x["_score"], reverse=True)
        return to_be_sorted

    docs_chunks_sorted = sort_chunks(gathered_docs)

    def sort_docs(docs: dict) -> List[dict]:
        as_list = list(docs.values())
        return sorted(as_list,  key=lambda x: x["chunks"][0]["_score"], reverse=True)

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
                    the_chunk["_source"][NeuralField.field_name]: the_chunk["_source"][NeuralField.field_content]
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
            cleaned = _clean_doc(d['doc']["_source"], doc_id=d['_id'])
            cleaned["_highlights"] = {
                d["chunks"][0]["_source"][NeuralField.field_name]: d["chunks"][0]["_source"][NeuralField.field_content]
            }
            cleaned["_score"] = d["chunks"][0]["_score"]
            simple_results.append(cleaned)
        return {"hits": simple_results[:result_count]}

    if simplified_format:
        return format_ordered_docs_simple(ordered_docs_w_chunks=completely_sorted)
    else:
        return format_ordered_docs_preserving(ordered_docs_w_chunks=completely_sorted, num_highlights=number_of_highlights)


def delete_index(config: Config, index_name):
    res = HttpRequests(config).delete(path=index_name)
    if index_name in get_cache():
        del get_cache()[index_name]
    return res


def _select_vectoriser(model_name: Union[str, MlModel]) -> Callable[[str], List[List[float]]]:
    """selects the vectorisation function based on the model's name"""
    if model_name == MlModel.bert or model_name == MlModel.clip:
        return functools.partial(s2_inference.vectorise, model_name)
    else:
        raise ValueError("neural_search: Unknown model selected: {}".format(model_name))


def _clean_doc(doc: dict, doc_id=None) -> dict:
    """clears neural search specific fields from the doc

    Args:
        doc: the doc to clean
        doc_id: if left as None, then the doc will be returned without the _id field

    Returns:

    """
    copied = doc.copy()
    if NeuralField.doc_chunk_relation in copied:
        del copied[NeuralField.doc_chunk_relation]
    if NeuralField.chunk_ids in copied:
        del copied[NeuralField.chunk_ids]
    if NeuralField.chunks in copied:
        del copied[NeuralField.chunks]
    if doc_id is not None:
        copied['_id'] = doc_id
    return copied


def _select_model_from_media_type(media_type: Union[MediaType, str]) -> Union[MlModel, str]:
    if media_type == MediaType.text:
        return MlModel.bert
    elif media_type == MediaType.image:
        return MlModel.clip
    else:
        raise ValueError("_select_model_from_media_type(): "
                         "Received unknown media type: {}".format(media_type))



