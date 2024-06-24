import json
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Sequence, Union
import semver


import jsonschema

import marqo.core.models.marqo_index as marqo_index
import marqo.exceptions as base_exceptions
from marqo.api.exceptions import (
    InvalidFieldNameError, InvalidArgError, InvalidDocumentIdError, DocTooLargeError)
from marqo.tensor_search import constants as tensor_search_constants
from marqo.tensor_search import enums, utils
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.delete_docs_objects import MqDeleteDocsRequest
from marqo.tensor_search.models.search import SearchContext
from marqo.tensor_search.models.mappings_object import (
    mappings_schema,
    multimodal_combination_mappings_schema,
    custom_vector_mappings_schema,
)
from marqo.core.models.marqo_index import *
from marqo.tensor_search.models.custom_vector_object import CustomVector
from marqo import marqo_docs


def validate_query(q: Optional[Union[dict, str]], search_method: Union[str, SearchMethod]) -> Optional[Union[dict, str]]:
    """
    Returns q if an error is not raised"""
    usage_ref = "\nSee query reference here: https://docs.marqo.ai/0.0.13/API-Reference/search/#query-q"

    if isinstance(q, str) or q is None:
        pass
    elif isinstance(q, dict):
        if search_method.upper() != SearchMethod.TENSOR:
            raise InvalidArgError(
                'Multi-query search is currently only supported for search_method="TENSOR" '
                f"\nReceived search_method `{search_method}`. {usage_ref}")
        if not len(q):
            raise InvalidArgError(
                "Multi-query search requires at least one query! Received empty dictionary. "
                f"{usage_ref}"
            )
        for k, v in q.items():
            base_invalid_kv_message = "Multi queries dictionaries must be <string>:<float> pairs. "
            if not isinstance(k, str):
                raise InvalidArgError(
                    f"{base_invalid_kv_message}Found key of type `{type(k)}` instead of string. Key=`{k}`"
                    f"{usage_ref}"
                )
            if not isinstance(v, (int, float)):
                raise InvalidArgError(
                    f"{base_invalid_kv_message}Found value of type `{type(v)}` instead of float. Value=`{v}`"
                    f" {usage_ref}"
                )
    else:
        raise InvalidArgError(
            f"'q' must be a 'string', a 'dict', or 'None' (if 'context' is provided)! Received q of type `{type(q)}`. "
            f"\nq=`{q}`"
            f"{usage_ref}"
        )
    return q


def validate_searchable_attributes(searchable_attributes: Optional[List[str]], search_method: SearchMethod):
    """Validate the searchable_attributes of an operation is not above the maximum number of attributes allowed.
    
    NOTE: There is only a maximum number of searchable attributes allowed for tensor search methods.

    """
    if search_method != SearchMethod.TENSOR:
        return

    maximum_searchable_attributes: Optional[str] = utils.read_env_vars_and_defaults(
        enums.EnvVars.MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES)
    if maximum_searchable_attributes is None:
        return

    if searchable_attributes is None:
        raise InvalidArgError(
            f"No searchable_attributes provided, but environment variable `MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES` is set."
        )

    if len(searchable_attributes) > int(maximum_searchable_attributes):
        raise InvalidArgError(
            f"Maximum searchable attributes (set via `MARQO_MAX_SEARCHABLE_TENSOR_ATTRIBUTES`) for tensor search is {maximum_searchable_attributes}, received {len(searchable_attributes)}."
        )


def validate_str_against_enum(value: Any, enum_class: Type[Enum], case_sensitive: bool = True):
    """Checks whether a value is found as the value of a str attribute of the
     given enum_class.

    Returns value if an error is not raised.
    """

    if case_sensitive:
        enum_values = set(item.value for item in enum_class)
        to_test_value = value
    else:
        enum_values = set(item.value.upper() for item in enum_class)
        to_test_value = value.upper()

    if to_test_value not in enum_values:
        raise ValueError(f"{value} is not a valid {enum_class.__name__}")
    return value


def list_types_valid(field_content: List) -> bool:
    """
    Verify all list elements are of the same type and that type is int, float or string.

    Returns:
        True if all elements are of the same type and that type is int, float or string
    """
    if len(field_content) == 0:
        return True
    list_type = type(field_content[0])
    if list_type not in [int, float, str]:
        return False
    for element in field_content:
        if type(element) != list_type:
            return False
    return True


def validate_list(field_content: List, is_non_tensor_field: bool):
    if type(field_content) is list and not list_types_valid(field_content):
        # if the field content is a list, it should only contain strings.
        raise InvalidArgError(
            f"Field content '{field_content}' "
            f"of type {type(field_content).__name__} is not of valid content type! "
            f"All list elements must be of the same type and that type must be int, float or string"
        )
    if not is_non_tensor_field:
        raise InvalidArgError(
            f"Field content '{field_content}' "
            f"of type {type(field_content).__name__} is not of valid content."
            f"Lists cannot be tensor fields"
        )
    return True


def validate_field_content(field_content: Any, is_non_tensor_field: bool) -> Any:
    """
    field: the field name of the field content. we need this to passed to validate_dict
    Returns
        field_content, if it is valid

    Raises:
        InvalidArgError if field_content is not acceptable
    """
    if type(field_content) in tensor_search_constants.ALLOWED_UNSTRUCTURED_FIELD_TYPES:
        if isinstance(field_content, list):
            validate_list(field_content, is_non_tensor_field)
        return field_content
    else:
        raise InvalidArgError(
            f"Field content `{field_content}` \n"
            f"of type `{type(field_content).__name__}` is not of valid content type!"
            f"Allowed content types: {[ty.__name__ for ty in tensor_search_constants.ALLOWED_UNSTRUCTURED_FIELD_TYPES]}"
        )


def validate_context(context: Optional[SearchContext], search_method: SearchMethod, query: Union[str, Dict[str, Any]]):
    """Validate the SearchContext.

    'validate_context' ensures that if the context is provided for a tensor search
    operation, the query must be a dictionary (not a str). 'context' 
    structure is validated internally.
    
    """
    if context is not None and search_method == SearchMethod.TENSOR and isinstance(query, str):
        raise InvalidArgError(
            f"Marqo received a query = `{query}` with type =`{type(query).__name__}` "
            f"and a parameter `context`.\n"  # do not return true {context} here as it might be huge.
            f"This is not supported as the context only works when the query is a dictionary."
            f"If you aim to search with your custom vectors, reformat the query as a dictionary.\n"
            f"Please check `{marqo_docs.search_context()}` for more information."
        )


def validate_boost(boost: Dict, search_method: Union[str, SearchMethod]):
    if boost is not None:
        further_info_message = ("\nRead about boost usage here: "
                                "https://docs.marqo.ai/0.0.13/API-Reference/search/#boost")
        for boost_attr in boost:
            try:
                validate_field_name(boost_attr)
            except InvalidFieldNameError as e:
                raise InvalidFieldNameError(f"Invalid boost dictionary. {e.message} {further_info_message}")
        if search_method != SearchMethod.TENSOR:
            # to be removed if boosting is implemented for lexical
            raise InvalidArgError(
                f'Boosting is only supported for search_method="TENSOR". '
                f'Received search_method={search_method}'
                f'{further_info_message}'
            )
        if not isinstance(boost, dict):
            raise InvalidArgError(
                f'Boost must be a dictionary. Instead received boost of value `{boost}`'
                f'{further_info_message}'
            )
        for k, v in boost.items():
            base_invalid_kv_message = (
                "Boost dictionaries have structure <attribute (string)>: <[weight (float), bias (float)]>\n")
            if not isinstance(k, str):
                raise InvalidArgError(
                    f'{base_invalid_kv_message}Found key of type `{type(k)}` instead of string. Key=`{k}`'
                    f"{further_info_message}"
                )
            if not isinstance(v, Sequence):
                raise InvalidArgError(
                    f'{base_invalid_kv_message}Found value of type `{type(v)}` instead of Array. Value=`{v}`'
                    f"{further_info_message}"
                )
            if len(v) not in [1, 2]:
                raise InvalidArgError(
                    f'{base_invalid_kv_message}An attribute boost must have a weight float and optional bias float. '
                    f'Instead received invalid boost `{v}`'
                    f"{further_info_message}"
                )
            for wb in v:
                if not isinstance(wb, (int, float)):
                    raise InvalidArgError(
                        f'{base_invalid_kv_message}An attribute boost must have a weight float and optional bias float. '
                        f'Instead received boost `{v}` with invalid member `{wb}` of type {type(wb)} '
                        f"{further_info_message}"
                    )
    return boost


def validate_field_name(field_name) -> str:
    """TODO:
        - length (remember the vector name will have the vector_prefix added to the front of field_name)
        - consider blanket "no double names starting with double underscore..."
    Args:
        field_name:

    returns field_name, if all validations pass

    Raises:
        InvalidFieldNameError
    """

    # TODO-Li Do we still have these restrictions for an unstructured index?
    if not field_name:
        raise InvalidFieldNameError("field name can't be empty! ")
    if not isinstance(field_name, str):
        raise InvalidFieldNameError("field name must be str!")
    if field_name.startswith(enums.TensorField.vector_prefix):
        raise InvalidFieldNameError(F"can't start field name with protected prefix {enums.TensorField.vector_prefix}."
                                    F" Error raised for field name: {field_name}")
    if field_name.startswith(enums.TensorField.chunks):
        raise InvalidFieldNameError(F"can't name field with protected field name {enums.TensorField.chunks}."
                                    F" Error raised for field name: {field_name}")
    char_validation = [(c, c not in tensor_search_constants.ILLEGAL_CUSTOMER_FIELD_NAME_CHARS)
                       for c in field_name]
    char_validation_failures = [c for c in char_validation if not c[1]]
    if char_validation_failures:
        raise InvalidFieldNameError(F"Illegal character '{char_validation_failures[0][0]}' "
                                    F"detected in field name {field_name}")
    if field_name not in enums.TensorField.__dict__.values():
        return field_name
    else:
        raise InvalidFieldNameError(f"field name can't be a protected field. Please rename this field: {field_name}")


def validate_unstructured_index_field_name(field_name) -> str:
    """TODO:
        - length (remember the vector name will have the vector_prefix added to the front of field_name)
        - consider blanket "no double names starting with double underscore..."
    Args:
        field_name:

    returns field_name, if all validations pass

    Raises:
        InvalidFieldNameError
    """

    # TODO-Li Do we still have these restrictions for an unstructured index?
    if not field_name:
        raise InvalidFieldNameError("field name can't be empty! ")
    if not isinstance(field_name, str):
        raise InvalidFieldNameError("field name must be str!")
    if field_name.startswith(enums.TensorField.vector_prefix):
        raise InvalidFieldNameError(F"can't start field name with protected prefix {enums.TensorField.vector_prefix}."
                                    F" Error raised for field name: {field_name}")
    if field_name.startswith(enums.TensorField.chunks):
        raise InvalidFieldNameError(F"can't name field with protected field name {enums.TensorField.chunks}."
                                    F" Error raised for field name: {field_name}")

    char_validation = [(c, c not in tensor_search_constants.ILLEGAL_CUSTOMER_FIELD_NAME_CHARS)
                       for c in field_name]
    char_validation_failures = [c for c in char_validation if not c[1]]
    if char_validation_failures:
        raise InvalidFieldNameError(F"Illegal character '{char_validation_failures[0][0]}' "
                                    F"detected in field name {field_name}")
    if field_name not in enums.TensorField.__dict__.values():
        return field_name
    else:
        raise InvalidFieldNameError(f"field name can't be a protected field. Please rename this field: {field_name}")


def validate_doc(doc: Dict) -> dict:
    """
    Args:
        doc: a document indexed by the client

    Raises:
        errors.InvalidArgError

    Returns
        doc if all validations pass
    """
    if not isinstance(doc, dict):
        raise InvalidArgError("Docs must be dicts")

    if len(doc) <= 0:
        raise InvalidArgError("Can't index an empty dict.")

    max_doc_size = utils.read_env_vars_and_defaults(var=enums.EnvVars.MARQO_MAX_DOC_BYTES)
    if max_doc_size is not None:
        try:
            serialized = json.dumps(doc)
        except TypeError as e:
            raise InvalidArgError(f"Unable to index document: it is not serializable! Document: `{doc}` ")
        if len(serialized) > int(max_doc_size):
            maybe_id = f" _id:`{doc['_id']}`" if '_id' in doc else ''
            raise DocTooLargeError(
                f"Document{maybe_id} with length `{len(serialized)}` exceeds "
                f"the allowed document size limit of [{max_doc_size}]."
            )
    return doc


def validate_id(_id: str):
    """Validates that an _id is ok

    Args:
        _id: to be validated

    Returns:
        _id, if it is acceptable
    """
    if not isinstance(_id, str):
        raise InvalidDocumentIdError(
            "Document _id must be a string type! "
            f"Received _id {_id} of type `{type(_id).__name__}`")
    if not _id:
        raise InvalidDocumentIdError("Document ID can't be empty")
    return _id


def validate_dict(field: str, field_content: Dict, is_non_tensor_field: bool, mappings: Dict,
                  index_model_dimensions: int = None, structured_field_type: FieldType = None,
                  marqo_index_version: semver.VersionInfo = None):
    """
    Args:
        field: the field name
        field_content: the field when it is a dict, especially used for multimodal tensor combination field
        structured_field_type: the type of the field (only for structured index).
        is_non_tensor_field: for multimodal tensor combination field, this should be True
        mappings: a dictionary to help validate the object field content
        index_model_dimensions: the dimensions of the model of the index. used to validate custom vector field.
        marqo_index_version: the version of the Marqo index
    Returns:
        Updated field_content dict or raise an error
    """
    marqo_index_version_lt_2_9_0 = marqo_index_version < semver.VersionInfo.parse("2.9.0")
    # If field is declared in mappings, it overwrites the default.
    if mappings and field in mappings:
        if mappings[field]["type"] == enums.MappingsObjectType.multimodal_combination:
            field_content = validate_multimodal_combination(field_content, is_non_tensor_field, mappings[field])
        elif mappings[field]["type"] == enums.MappingsObjectType.custom_vector:
            field_content = validate_custom_vector(field_content, is_non_tensor_field, index_model_dimensions)
        else:
            raise InvalidArgError(
                f"The field `{field}` is of invalid type in the `mappings` parameter. The only object field type supported "
                f"is `custom_vector`. However, the `mappings` provided is: {mappings}. Please change the "
                f"type of {field}. "
                f"See `{marqo_docs.mappings()}` for more info on object fields. "
            )
    else:
        # Use type declared in structured_field_type if not in mappings
        if structured_field_type:
            if structured_field_type == FieldType.MultimodalCombination:
                field_content = validate_multimodal_combination(field_content, is_non_tensor_field, mappings[field])
            elif structured_field_type == FieldType.CustomVector:
                field_content = validate_custom_vector(field_content, is_non_tensor_field, index_model_dimensions)
            elif structured_field_type in [FieldType.MapFloat, FieldType.MapInt, FieldType.MapDouble, FieldType.MapLong]:
                field_content = validate_map_numeric_field(field_content)
            else:
                raise InvalidArgError(
                    f"The field {field} is of type `{structured_field_type}`, which is not a valid object field type. "
                    f"The only object field type supported "
                    f"is `custom_vector`. Please change the type of {field} to use it as an object field. "
                    f"See `{marqo_docs.mappings()}` for more info on object fields. "
                )
        else:
            # Index is unstructured, check if it is a map numeric field
            if marqo_index_version_lt_2_9_0:
                raise InvalidArgError(
                    f"Unsupported map field. "
                    f"In Marqo versions prior to 2.9.0, only custom vector fields are supported as map fields "
                    f"and must be declared in the `mappings` parameter (See `{marqo_docs.mappings()}). "
                    f"Only indexes created with Marqo 2.9.0 or above support map numerical fields."
                )
            
            field_content = validate_map_numeric_field(field_content)

    return field_content

def validate_map_numeric_field(field_content):
     """
     Validates the field content if it is a map field (dict)
     Args:
         field_content: the field content
     Returns:
         field_content if the validation passes
     """

     # Validate that the field content is a dict
     if not isinstance(field_content, dict):
         raise InvalidArgError(
             f"The field content `{field_content}` is of type `{type(field_content).__name__}`, which is not a valid type for a map field."
             f"A map field must be a dictionary."
         )

     # Validate that the dict is only of one level
     if any(isinstance(v, dict) for v in field_content.values()):
         raise InvalidArgError(
             "Nested dictionaries are not allowed in map fields. Each value must be a single int, float, or double."
         )

     # Validate that the values of the dict are int, float, or double
     for key, value in field_content.items():
         if not isinstance(key, str):
             raise InvalidArgError(
                 f"Key `{key}` in map field is not a string. All keys must be strings."
             )
         if not isinstance(value, (int, float)):
             raise InvalidArgError(
                 f"Value `{value}` for key `{key}` in map field is not of type int or float."
             )

     return field_content

def validate_multimodal_combination(field_content, is_non_tensor_field, field_mapping):
    """
    Validates the field content if it is a multimodal combination field (dict)
    Args:
        field_content: the field content
        is_non_tensor_field: whether this is a non-tensor-field
        field_mapping: the mapping to help validate this field content

    Returns:
        The field content
    """
    if len(field_content) < 1:
        raise InvalidArgError(
            f"The multimodal_combination field `{field_content}` is an empty dictionary."
            f"This is not a valid format of field content."
            f"If you aim to use multimodal_combination, it must contain at least 1 field. "
            f"please check `{marqo_docs.multimodal_combination_object()}` for more info.")

    for key, value in field_content.items():
        if not ((type(key) in tensor_search_constants.ALLOWED_MULTIMODAL_FIELD_TYPES) and (
                type(value) in tensor_search_constants.ALLOWED_MULTIMODAL_FIELD_TYPES)):
            raise InvalidArgError(
                f"Multimodal-combination field content `{key}:{value}` \n  "
                f"of type `{type(key).__name__} : {type(value).__name__}` is not of valid content type (one of {tensor_search_constants.ALLOWED_MULTIMODAL_FIELD_TYPES})."
            )

        if not key in field_mapping["weights"]:
            raise InvalidArgError(
                f"Multimodal-combination field content `{key}:{value}` \n  "
                f"is not in the multimodal_field mappings weights `{field_mapping['weights']}`. Each sub_field requires a weight."
                f"Please add `{key}` to the mappings."
                f"Please check `{marqo_docs.multimodal_combination_object()}` for more info.")

    if is_non_tensor_field:
        raise InvalidArgError(
            f"Field content `{field_content}` \n  "
            f"of type `{type(field_content).__name__}` is the content for a multimodal_combination."
            f"It must be a tensor field. Add this field to `tensorFields` or "
            f"add it as a normal field to fix this problem."
        )
    return field_content


def validate_custom_vector(field_content: dict, is_non_tensor_field: bool, index_model_dimensions: int):
    """
    Validates the field content if it is a custom vector field (dict)
    Args:
        field_content: the field content
        is_non_tensor_field: whether this is a non-tensor-field
        index_model_dimensions: the `dimensions` property of the index to be added to
    Returns:
        field_content if the validation passes
        "content" key will be added to the field_content as empty string if it is not provided.
    """

    validated_custom_vector = CustomVector(
        dict_data=field_content,
        dimension=index_model_dimensions,
        is_non_tensor_field=is_non_tensor_field
    )
    return validated_custom_vector.to_dict()


def validate_mappings_object(
        mappings_object: Dict,
        structured_marqo_index: Optional[marqo_index.StructuredMarqoIndex] = None
):
    """validates the mappings object.
    Args:
        mappings_object: the mappings object to validate
        structured_marqo_index: Optional StructuredMarqoIndex object. If provided, the validation ensures the mappings are valid
        for this index
    Returns
        The given mappings object if validation has passed

    Raises an InvalidArgError if the settings object is badly formatted
    """
    try:
        jsonschema.validate(instance=mappings_object, schema=mappings_schema)
        for field_name, config in mappings_object.items():
            validate_field_name(field_name)
            if config["type"] == enums.MappingsObjectType.multimodal_combination:
                validate_multimodal_combination_mappings_object(config)
                if structured_marqo_index is not None:
                    if (
                            field_name not in structured_marqo_index.field_map or
                            structured_marqo_index.field_map[
                                field_name
                            ].type != marqo_index.FieldType.MultimodalCombination
                    ):
                        raise jsonschema.ValidationError(
                            f'Invalid mapping {field_name}. Index has no multimodal combination field {field_name}'
                        )
                    dependent_fields = structured_marqo_index.field_map[field_name].dependent_fields
                    weights: dict = config['weights']
                    for field in weights:
                        if field not in dependent_fields:
                            raise jsonschema.ValidationError(
                                f'Invalid mapping {field_name}. '
                                f'Field {field} is not a dependent field of {field_name}'
                            )

            elif config["type"] == enums.MappingsObjectType.custom_vector:
                validate_custom_vector_mappings_object(config)
                # TODO: add validation for custom vector structured/unstructured here


        return mappings_object
    except jsonschema.ValidationError as e:
        raise InvalidArgError(
            f"Error validating mappings object. Reason: {str(e)}. "
            f" Read about the mappings object here: `{marqo_docs.mappings()}`"
        )
    
def validate_multimodal_combination_mappings_object(mappings_object: Dict):
    """Validates the multimodal mappings object

    Args:
        multimodal_mappings:

    Returns:
        The original object, if it passes validation
    Raises InvalidArgError if the object is badly formatted
    """
    try:
        jsonschema.validate(instance=mappings_object, schema=multimodal_combination_mappings_schema)
    except jsonschema.ValidationError as e:
        raise InvalidArgError(
            f"Error validating multimodal combination mappings object. Reason: \n{str(e)}"
            f"\n Read about the mappings object here: `{marqo_docs.mappings()}`"
        )

    # TODO: compare, this might be wrong for v2.
    # TODO: Move this validation into schema in mappings_object
    for child_field, weight in mappings_object["weights"].items():
        # TODO: We may need to validate field name of child field.
        if type(child_field) not in tensor_search_constants.ALLOWED_MULTIMODAL_FIELD_TYPES:
            raise InvalidArgError(
                f"The multimodal_combination mapping `{mappings_object}` has an invalid child_field `{child_field}` of type `{type(child_field).__name__}`."
                f"In multimodal_combination fields, it must be a string."
                f"Please check `{marqo_docs.multimodal_combination_object()}` for more info."
            )

        if not isinstance(weight, (float, int)):
            raise InvalidArgError(
                f"The multimodal_combination mapping `{mappings_object}` has an invalid weight `{weight}` of type `{type(weight).__name__}`."
                f"In multimodal_combination fields, weight must be an int or float."
                f"Please check `{marqo_docs.multimodal_combination_object()}` for more info."
            )

    return mappings_object


def validate_custom_vector_mappings_object(mappings_object: Dict):
    """Validates the custom vector mappings object
    Args:
        mappings_object:
    Returns:
        The original object, if it passes validation
    Raises InvalidArgError if the object is badly formatted
    Example custom vector mappings must look exactly like this:
    "my_custom_vector_field": {
        "type": "custom_vector"
    }
    """
    try:
        jsonschema.validate(instance=mappings_object, schema=custom_vector_mappings_schema)
    except jsonschema.ValidationError as e:
        raise InvalidArgError(
            f"Error validating custom vector mappings object. Reason: \n{str(e)}"
            f"\n Read about the mappings object here: `{marqo_docs.mappings()}`"
        )

    return mappings_object


def validate_delete_docs_request(delete_request: MqDeleteDocsRequest, max_delete_docs_count: int):
    """Validates a delete docs request from the user.

    Args:
        delete_request: A deletion request from the user
        max_delete_docs_count: the maximum allowed docs to delete. Should be
            set by the env var MARQO_MAX_DELETE_DOCS_COUNT
    Returns:
        del_request, if nothing is raised
    """
    if not isinstance(delete_request, MqDeleteDocsRequest):
        raise RuntimeError("Deletion request must be a MqDeleteDocsRequest object")

    if not isinstance(max_delete_docs_count, int):
        raise RuntimeError("max_delete_docs_count must be an int!")

    if not delete_request.document_ids:
        # TODO: refactor doc_ids to use the correct API parameter name (documentIds)
        raise InvalidDocumentIdError("doc_ids can't be empty!")

    if not isinstance(delete_request.document_ids, Sequence) or isinstance(delete_request.document_ids, str):
        raise InvalidArgError("documentIds param must be an array of strings.")

    if (len(delete_request.document_ids) > max_delete_docs_count) and max_delete_docs_count is not None:
        raise InvalidArgError(
            f"The number of documentIds to delete `{len(delete_request.document_ids)}` is "
            f"greater than the limit `{max_delete_docs_count}` set by the env var "
            f"`{enums.EnvVars.MARQO_MAX_DELETE_DOCS_COUNT}`. ")

    for _id in delete_request.document_ids:
        validate_id(_id)

    return delete_request
