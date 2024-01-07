import jsonschema

from typing import Dict, Optional, List
from marqo import errors
from marqo.tensor_search.models.mappings_object import mappings_schema, multimodal_combination_schema
from marqo.errors import (
    InvalidFieldNameError, InvalidArgError, InvalidDocumentIdError, DocTooLargeError)
from marqo.tensor_search import enums
from marqo.core.models.marqo_index import validate_field_name as common_validate_name

_FILTER_STRING_BOOL_VALUES = ["true", "false"]
_RESERVED_FIELD_SUBSTRING = "::"
_SUPPORTED_FIELD_CONTENT_TYPES = [str, int, float, bool, list]


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
        raise InvalidArgError(
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
        raise InvalidArgError(
            f"Error validating multimodal combination mappings object. Reason: \n{str(e)}"
            f"\n Read about the mappings object here: https://docs.marqo.ai/1.4.0/API-Reference/Documents/mappings/"
        )


def validate_field_name(field_name: str) -> None:
    """Validate the field name.

    We reuse the validation function from structured index, but add some extra validations for unstructured index."""
    try:
        common_validate_name(field_name)
    except ValueError as e:
        raise errors.InvalidFieldNameError(e)
    # Some extra validations for unstructured index
    if _RESERVED_FIELD_SUBSTRING in field_name:
        raise errors.InvalidFieldNameError(
            f"Field name {field_name} contains the reserved substring {_RESERVED_FIELD_SUBSTRING}. This is a "
            f"reserved substring and cannot be used in field names for unstructured marqo index."
        )


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
                raise InvalidArgError(
                    f"Multimodal subfields must be strings representing text or image pointer, "
                    f"received {sub_field}:{sub_content}, which is of type {type(sub_content).__name__}")


def _validate_conflicts_fields(multimodal_fields: List[str], doc: Dict):
    mappings_fields = set(multimodal_fields)
    doc_fields = set(doc.keys())
    if mappings_fields.intersection(doc_fields):
        raise InvalidArgError(
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

