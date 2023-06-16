"""Communication with Marqo's persistence and search layer (OpenSearch)"""
import copy
import json
import typing
from marqo.tensor_search.models.index_info import IndexInfo
# client-specific modules - we may want to replace these:
from marqo._httprequests import HttpRequests
from marqo.config import Config
from marqo.errors import MarqoError
from marqo.tensor_search import validation, constants, enums
from marqo.tensor_search import utils
from marqo import errors
#
from typing import Iterable, List, Union, Optional, Tuple, Dict
from marqo.tensor_search.index_meta_cache import get_cache
from marqo.tensor_search.index_meta_cache import get_index_info as get_cached_index_info
import pprint


def get_index_info(config: Config, index_name: str) -> IndexInfo:
    """Gets useful information about the index. Also updates the IndexInfo cache

    Args:
        config:
        index_name:

    Returns:
        IndexInfo of the index

    Raises:
        NonTensorIndexError, if the index's mapping doesn't conform to a Tensor Search index

    """
    res = HttpRequests(config).get(path=F"{index_name}/_mapping")

    if not (index_name in res and "mappings" in res[index_name]
            and "_meta" in res[index_name]["mappings"]):
        raise errors.NonTensorIndexError(
            f"Error retrieving index info for index {index_name}")

    if "model" in res[index_name]["mappings"]["_meta"]:
        model_name = res[index_name]["mappings"]["_meta"]["model"]
    else:
        raise errors.NonTensorIndexError(
            "get_index_info: couldn't identify embedding model name "
            F"in index mappings! Mapping: {res}")

    if "index_settings" in res[index_name]["mappings"]["_meta"]:
        index_settings = res[index_name]["mappings"]["_meta"]["index_settings"]
    else:
        raise errors.NonTensorIndexError(
            "get_index_info: couldn't identify index_settings "
            F"in index mappings! Mapping: {res}")

    index_properties = res[index_name]["mappings"]["properties"]

    index_info = IndexInfo(model_name=model_name, properties=index_properties,
                           index_settings=index_settings)
    get_cache()[index_name] = index_info
    return index_info


def add_customer_field_properties(config: Config, index_name: str,
                                  customer_field_names: Iterable[Tuple[str, enums.OpenSearchDataType]],
                                  model_properties: dict, multimodal_combination_fields: Dict[str, Iterable[Tuple[str, enums.OpenSearchDataType]]]):
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
    existing_info = get_cached_index_info(config=config, index_name=index_name)

    # check if there is multimodal fie;ds and convert the fields name to a list with the same
    # format of customer_field_names
    knn_field_names = copy.deepcopy(customer_field_names)
    if len(multimodal_combination_fields) > 0:
        multimodal_customer_field_names = set([(field_name, "_") for field_name in list(multimodal_combination_fields)])
        knn_field_names = knn_field_names.union(multimodal_customer_field_names)

    body = {
        "properties": {
            enums.TensorField.chunks: {
                "type": "nested",
                "properties": {}
            }
        }
    }
    new_index_properties = existing_info.properties.copy()

    # copy fields to the chunk for prefiltering. If it is text, convert it to a keyword type to save space
    # if it's not text, ignore it, and leave it up to OpenSearch (e.g: if it's a number)
    for field_name in customer_field_names:
        if field_name[1] == enums.OpenSearchDataType.text \
                or field_name[1] == enums.OpenSearchDataType.keyword:
            body["properties"][enums.TensorField.chunks]["properties"][validation.validate_field_name(field_name[0])] = {
                "type": enums.OpenSearchDataType.keyword,
                "ignore_above": 32766  # this is the Marqo-OS bytes limit
            }

    for field_name in list(multimodal_combination_fields):
        body["properties"][enums.TensorField.chunks]["properties"][
            validation.validate_field_name(field_name)] = {"properties":{
            sub_field[0]:{
                "type": enums.OpenSearchDataType.keyword,
                "ignore_above": 32766  # this is the Marqo-OS bytes limit
            } for sub_field in multimodal_combination_fields[field_name]
        },
    }

    mapping_res = HttpRequests(config).put(path=F"{index_name}/_mapping", body=json.dumps(body))

    merged_chunk_properties = {
        **existing_info.properties[enums.TensorField.chunks]["properties"],
        **body["properties"][enums.TensorField.chunks]["properties"]
    }
    new_index_properties[enums.TensorField.chunks]["properties"] = merged_chunk_properties

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


    for multimodal_field, child_fields in multimodal_combination_fields.items():
        # update the new multimodal_field if it's not in it
        if multimodal_field not in new_index_properties:
            new_index_properties[validation.validate_field_name(multimodal_field)] = \
                {"properties": {validation.validate_field_name(child_field_name): {"type":child_type}
                 for child_field_name, child_type in child_fields}}
        # update the new child fields if the multimodal_field already in it
        else:
            for child_field_name, child_type in child_fields:
                new_index_properties[validation.validate_field_name(multimodal_field)]["properties"][child_field_name] = {"type":child_type}

    get_cache()[index_name] = IndexInfo(
        model_name=existing_info.model_name,
        properties=new_index_properties,
        index_settings=existing_info.index_settings.copy()
    )
    return mapping_res


def get_cluster_indices(config: Config):
    """Gets the name of all indices, excluding system indices"""
    res = HttpRequests(config).get(path="_aliases")
    return _remove_system_indices(res.keys())


def _remove_system_indices(index_names: typing.Iterable) -> set:
    """Removes system indices from the set of indices

    Args:
        index_names: all names of indices, retrieved from an index

    Returns:
        A set of indices that don't include system indices
    """
    indices = set(index_names)
    indices_no_protected_prefixes = {index_name for index_name in indices
                                     if not any([index_name.startswith(prefix) for prefix in
                                                constants.INDEX_NAME_PREFIXES_TO_IGNORE])}
    indices_no_system_names = indices_no_protected_prefixes - constants.INDEX_NAMES_TO_IGNORE
    return indices_no_system_names
