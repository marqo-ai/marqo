"""Communication with Neural Search's persistence and search layer (OpenSearch)"""
import json
from marqo.neural_search.models.index_info import IndexInfo
# client-specific modules - we may want to replace these:
from marqo._httprequests import HttpRequests
from marqo.config import Config
from marqo.errors import MarqoError
from marqo.neural_search import validation, constants, enums
from marqo.neural_search import utils
from marqo import errors
#
from typing import Iterable, List, Union, Optional, Tuple
from marqo.neural_search.index_meta_cache import get_cache


def get_index_info(config: Config, index_name: str) -> IndexInfo:
    """Gets useful information about the index. Also updates the IndexInfo cache

    Args:
        config:
        index_name:

    Returns:
        IndexInfo of the index

    Raises:
        NonNeuralIndexError, if the index's mapping doesn't conform to a Neural Search index

    """
    res = HttpRequests(config).get(path=F"{index_name}/_mapping")

    if not (index_name in res and "mappings" in res[index_name]
            and "_meta" in res[index_name]["mappings"]):
        raise errors.NonNeuralIndexError(
            f"Error retrieving index info for index {index_name}")

    if "model" in res[index_name]["mappings"]["_meta"]:
        model_name = res[index_name]["mappings"]["_meta"]["model"]
    else:
        raise errors.NonNeuralIndexError(
            "get_index_info: couldn't identify embedding model name "
            F"in index mappings! Mapping: {res}")

    if "neural_settings" in res[index_name]["mappings"]["_meta"]:
        neural_settings = res[index_name]["mappings"]["_meta"]["neural_settings"]
    else:
        raise errors.NonNeuralIndexError(
            "get_index_info: couldn't identify neural_settings "
            F"in index mappings! Mapping: {res}")

    index_properties = res[index_name]["mappings"]["properties"]

    index_info = IndexInfo(model_name=model_name, properties=index_properties,
                           neural_settings=neural_settings)
    get_cache()[index_name] = index_info
    return index_info


def add_customer_field_properties(config: Config, index_name: str,
                                  customer_field_names: Iterable[Tuple[str, enums.OpenSearchDataType]],
                                  model_properties: dict):
    """Adds new customer fields to index mapping.

    Pushes the updated mapping to OpenSearch, and updates the local cache.

    Args:
        config:
        index_name:
        customer_field_names: list of 2-tuples. The first elem in the tuple is
            the new fieldnames the customers have made. The second elem is the
            inferred OpenSearch data type.
        model_properties: properties of the machine learning model

    Returns:
        HTTP Response
    """
    body = {
        "properties": {
            enums.NeuralField.chunks: {
                "type": "nested",
                "properties": {
                    validation.validate_vector_name(
                        utils.generate_vector_name(field_name[0])): {
                        "type": "knn_vector",
                        "dimension": model_properties["dimensions"],
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",  # or try Lucene
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    } for field_name in customer_field_names
                }
            }
        }
    }

    existing_info = get_cache()[index_name]
    new_index_properties = existing_info.properties.copy()

    # copy fields to the chunk for prefiltering. If it is text, convert it to a keyword type to save space
    # if it's not text, ignore it, and leave it up to OpenSearch (e.g: if it's a number)
    for field_name in customer_field_names:
        nested_non_vector_field_type = enums.OpenSearchDataType.keyword if field_name[1] == enums.OpenSearchDataType.text else field_name[1]
        if nested_non_vector_field_type == enums.OpenSearchDataType.text \
                or nested_non_vector_field_type == enums.OpenSearchDataType.keyword:
            body["properties"][enums.NeuralField.chunks]["properties"][validation.validate_field_name(field_name[0])] = {
                "type": enums.OpenSearchDataType.keyword
            }

    mapping_res = HttpRequests(config).put(path=F"{index_name}/_mapping", body=json.dumps(body))

    merged_chunk_properties = {
        **existing_info.properties[enums.NeuralField.chunks]["properties"],
        **body["properties"][enums.NeuralField.chunks]["properties"]
    }
    new_index_properties[enums.NeuralField.chunks]["properties"] = merged_chunk_properties

    # Save newly created fields to document-level so that it is searchable by lexical search
    # These will be undefined, and we let OpenSearch define them, the next
    #   time they're retrieved from the cache
    existing_properties = set(existing_info.get_text_properties())
    applying_properties = {field[0] for field in customer_field_names}
    app_type_mapping = {field: field_type for field, field_type in customer_field_names}
    new_properties = applying_properties - existing_properties
    for new_prop in new_properties:
        type_to_set = app_type_mapping[new_prop] if app_type_mapping[new_prop] == enums.OpenSearchDataType.text \
                        else enums.OpenSearchDataType.to_be_defined
        new_index_properties[validation.validate_field_name(new_prop)] = {
            "type": type_to_set
        }

    get_cache()[index_name] = IndexInfo(
        model_name=existing_info.model_name,
        properties=new_index_properties,
        neural_settings=existing_info.neural_settings.copy()
    )
    return mapping_res


def get_cluster_indices(config: Config):
    """Gets the name of all indices"""
    res = HttpRequests(config).get(path="_aliases")
    indices = set(res.keys())
    relevant_indices = indices - constants.INDEX_NAMES_TO_IGNORE
    return relevant_indices
