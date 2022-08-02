"""Communication with Neural Search's persistence and search layer (OpenSearch)"""
import json
import pprint
from marqo.neural_search.models.index_info import IndexInfo
# client-specific modules - we may want to replace these:
from marqo._httprequests import HttpRequests
from marqo.config import Config
from marqo.errors import MarqoError
from marqo.neural_search import validation, constants, enums
from marqo.neural_search import utils
from marqo import errors
#
from typing import Iterable, List, Union, Optional
from marqo.neural_search.index_meta_cache import get_cache


def get_index_info(config: Config, index_name: str) -> IndexInfo:
    """Gets useful information about the index. Also updates the IndexInfo cache

    Args:
        config:
        index_name:

    Returns:
        IndexInfo of the index

    Raises:
        MarqoError, if the index's mapping doesn't conform to a Neural Search index

    """
    res = HttpRequests(config).get(path=F"{index_name}/_mapping")

    if not (index_name in res and "mappings" in res[index_name]
            and "_meta" in res[index_name]["mappings"]):
        raise errors.MarqoNonNeuralIndexError(
            f"Error retrieving index info for index {index_name}")

    if "model" in res[index_name]["mappings"]["_meta"]:
        model_name = res[index_name]["mappings"]["_meta"]["model"]
    else:
        raise errors.MarqoNonNeuralIndexError(
            "get_index_info: couldn't identify embedding model name "
            F"in index mappings! Mapping: {res}")

    if "neural_settings" in res[index_name]["mappings"]["_meta"]:
        neural_settings = res[index_name]["mappings"]["_meta"]["neural_settings"]
    else:
        raise errors.MarqoNonNeuralIndexError(
            "get_index_info: couldn't identify neural_settings "
            F"in index mappings! Mapping: {res}")

    index_properties = res[index_name]["mappings"]["properties"]

    index_info = IndexInfo(model_name=model_name, properties=index_properties,
                           neural_settings=neural_settings)
    get_cache()[index_name] = index_info
    return index_info


def add_customer_field_properties(config: Config, index_name: str, customer_field_names: Iterable[str], model_properties: dict):
    """Adds new customer fields to index mapping.

    Args:
        config:
        index_name:
        customer_field_names: the new fieldnames the customers have made
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
                        utils.generate_vector_name(field_name)): {
                        "type": "knn_vector",
                        "dimension": model_properties["dimensions"],
                        "method": {
                            "name": "hnsw",
                            "space_type": "innerproduct",
                            "engine": "nmslib",
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

    for field_name in customer_field_names:
        body["properties"][validation.validate_field_name(field_name)] = {
            "type": "text"
        }
        new_index_properties[validation.validate_field_name(field_name)] = {
            "type": "text"
        }

    merged_chunk_properties = {
        **existing_info.properties[enums.NeuralField.chunks]["properties"],
        **body["properties"][enums.NeuralField.chunks]["properties"]
    }
    new_index_properties[enums.NeuralField.chunks]["properties"] = merged_chunk_properties
    get_cache()[index_name] = IndexInfo(
        model_name=existing_info.model_name,
        properties=new_index_properties,
        neural_settings=existing_info.neural_settings.copy()
    )
    mapping_res = HttpRequests(config).put(path=F"{index_name}/_mapping", body=json.dumps(body))
    return mapping_res


def get_cluster_indices(config: Config):
    """Gets the name of all indices"""
    res = HttpRequests(config).get(path="_aliases")
    indices = set(res.keys())
    relevant_indices = indices - constants.INDEX_NAMES_TO_IGNORE
    return relevant_indices
