from typing import Dict, Optional, List

import jsonschema

from marqo.api import exceptions as errors
from marqo.tensor_search import enums
from marqo.tensor_search.models.mappings_object import mappings_schema, multimodal_combination_schema

_FILTER_STRING_BOOL_VALUES = ["true", "false"]
_RESERVED_FIELD_SUBSTRING = "::"
_SUPPORTED_FIELD_CONTENT_TYPES = [str, int, float, bool, list]
_ILLEGAL_CUSTOMER_FIELD_NAME_CHARS = {'.', '/', '\n'}


def validate_mappings_object_format(mappings: Dict) -> None:
    """Validate the format of the mappings object
    Args: mappings: the mappings object
    Raises InvalidArgError if the object is badly formatted, and this should terminate the add_document process
    """
    try:
        jsonschema.validate(instance=mappings, schema=mappings_schema)
        for field_name, configuration in mappings.items():
            if configuration["type"] == enums.MappingsObjectType.multimodal_combination:
                _validate_multimodal_combination_field_name(field_name)
                _validate_multimodal_combination_configuration_format(configuration)
    except jsonschema.ValidationError as e:
        raise errors.InvalidArgError(
            f"Error validating mappings object. Reason: {str(e)}. "
            f"Read about the mappings object here: https://docs.marqo.ai/0.0.15/API-Reference/mappings/"
        )


def _validate_multimodal_combination_field_name(multimodal_field_name: str):
    validate_field_name(multimodal_field_name)
    # This error will never be raised because _id can't be a tensor_fields
    # TODO check if there are other validations needed for multimodal_field_name
    if multimodal_field_name == "_id":
        raise errors.InvalidArgError("multimodal field name can not be _id")


def _validate_multimodal_combination_configuration_format(configuration: Dict):
    try:
        jsonschema.validate(instance=configuration, schema=multimodal_combination_schema)
    except jsonschema.ValidationError as e:
        raise errors.InvalidArgError(
            f"Error validating multimodal combination mappings object. Reason: \n{str(e)}"
            f"\n Read about the mappings object here: https://docs.marqo.ai/1.4.0/API-Reference/Documents/mappings/"
        )


def _validate_custom_vector_mappings_object(configuration: Dict):
    pass


def validate_field_name(field_name: str):
    if not field_name:
        raise errors.InvalidFieldNameError("field name can't be empty! ")
    if not isinstance(field_name, str):
        raise errors.InvalidFieldNameError("field name must be str!")
    if field_name.startswith(enums.TensorField.vector_prefix):
        raise errors.InvalidFieldNameError(
            f"can't start field name with protected prefix {enums.TensorField.vector_prefix}."
            f" Error raised for field name: {field_name}")
    if field_name.startswith(enums.TensorField.chunks):
        raise errors.InvalidFieldNameError(f"can't name field with protected field name {enums.TensorField.chunks}."
                                           f" Error raised for field name: {field_name}")
    char_validation = [(c, c not in _ILLEGAL_CUSTOMER_FIELD_NAME_CHARS)
                       for c in field_name]
    char_validation_failures = [c for c in char_validation if not c[1]]
    if char_validation_failures:
        raise errors.InvalidFieldNameError(F"Illegal character '{char_validation_failures[0][0]}' "
                                           F"detected in field name {field_name}")
    if field_name in enums.TensorField.__dict__.values():
        raise errors.InvalidFieldNameError(
            f"field name can't be a protected field. Please rename this field: {field_name}")
    if _RESERVED_FIELD_SUBSTRING in field_name:
        raise errors.InvalidFieldNameError(
            f"Field name {field_name} contains the reserved substring {_RESERVED_FIELD_SUBSTRING}. This is a "
            f"reserved substring and cannot be used in field names for unstructured marqo index."
        )
    return


def validate_coupling_of_mappings_and_doc(doc: Dict, mappings: Dict, multimodal_sub_fields: List):
    """Validate the coupling of mappings object and doc"""
    if not mappings:
        return

    multimodal_fields = [field_name for field_name, configuration in mappings.items()
                         if configuration["type"] == enums.MappingsObjectType.multimodal_combination]

    if multimodal_fields:
        _validate_conflicts_fields(multimodal_fields, doc)
        _validate_multimodal_sub_fields_content(doc, multimodal_sub_fields)


def _validate_multimodal_sub_fields_content(doc: Dict, multimodal_sub_fields: List):
    for sub_field in multimodal_sub_fields:
        if sub_field in doc:
            sub_content = doc[sub_field]
            if not isinstance(sub_content, str):
                raise errors.InvalidArgError(
                    f"Multimodal subfields must be strings representing text or image pointer, "
                    f"received {sub_field}:{sub_content}, which is of type {type(sub_content).__name__}")


def _validate_conflicts_fields(multimodal_fields: List[str], doc: Dict):
    mappings_fields = set(multimodal_fields)
    doc_fields = set(doc.keys())
    if mappings_fields.intersection(doc_fields):
        raise errors.InvalidArgError(
            f"Document and mappings object have conflicting fields: {mappings_fields.intersection(doc_fields)}")


def validate_tensor_fields(tensor_fields: Optional[List[str]]) -> None:
    """Validate the tensor fields
    Args:
        tensor_fields: the tensor fields
    Raises InvalidArgError if the tensor fields are invalid, and this should terminate the add_document process
    """
    if tensor_fields is None:
        raise errors.BadRequestError("tensor_fields must be explicitly provided as a list for unstructured index. "
                                     "If you don't want to vectorise any field, please provide an empty list [].")
    if "_id" in tensor_fields:
        raise errors.BadRequestError(message="`_id` field cannot be a tensor field.")
