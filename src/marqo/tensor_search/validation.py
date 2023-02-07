import json
import pprint
import typing
from marqo.tensor_search import constants
from marqo.tensor_search import enums, utils
from typing import Iterable, Container, Union
from marqo.errors import (
    MarqoError, InvalidFieldNameError, InvalidArgError, InternalError,
    InvalidDocumentIdError, DocTooLargeError, InvalidIndexNameError)
from marqo.tensor_search.enums import TensorField, SearchMethod
from marqo.tensor_search import constants
from typing import Any, Type
import inspect
from enum import Enum


def validate_query(q: Union[dict, str], search_method: Union[str, SearchMethod]):
    """
    Returns q if an error is not raised"""
    if isinstance(q, dict):
        if search_method.upper() != SearchMethod.TENSOR:
            raise ValueError(
                "Multi-query search is currently only supported for search_method=TENSOR! "
                f"\nReceived search_method `{search_method}`")
        for k, v in q.items():
            base_invalid_kv_message = (
                "Multi queries dictionaries must be <string>:<float> pairs. See usage here: "
                ""  # FIXME: add link
            )
            if not isinstance(k, str):
                raise ValueError(f"{base_invalid_kv_message}\n"
                                 f"    Found key of type `{type(k)}` instead of string. Key=`{k}`")
            if not isinstance(v, (int, float)):
                raise ValueError(f"{base_invalid_kv_message}\n"
                                 f"    Found value of type `{type(v)}` instead of float. Value=`{v}`")
    elif not isinstance(q, str):
        raise ValueError(
            f"q must be a string or dict! Received q of type `{type(q)}`. "
            f"\nq=`{q}`")
    return q


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


def validate_field_content(field_content: typing.Any) -> typing.Any:
    """

    Returns
        field_content, if it is valid

    Raises:
        InvalidArgError if field_content is not acceptable
    """
    if type(field_content) in constants.ALLOWED_CUSTOMER_FIELD_TYPES:
        return field_content
    else:
        raise InvalidArgError(
            f"Field content `{field_content}` \n"
            f"of type `{type(field_content).__name__}` is not of valid content type!"
            f"Allowed content types: {[ty.__name__ for ty in constants.ALLOWED_CUSTOMER_FIELD_TYPES]}"
        )


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
    char_validation = [(c, c not in constants.ILLEGAL_CUSTOMER_FIELD_NAME_CHARS)
                        for c in field_name]
    char_validation_failures = [c for c in char_validation if not c[1]]
    if char_validation_failures:
        raise InvalidFieldNameError(F"Illegal character '{char_validation_failures[0][0]}' "
                               F"detected in field name {field_name}")
    if field_name not in enums.TensorField.__dict__.values():
        return field_name
    else:
        raise InvalidFieldNameError(f"field name can't be a protected field. Please rename this field: {field_name}")


def validate_doc(doc: dict) -> dict:
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


def validate_vector_name(name: str):
    """Checks that the vector name is valid.
    It should have the form __vector_{customer field name}

    Raises:
        errors.InternalError, as vector names are an internal concern and
            should be hidden from the end user
    """
    if not isinstance(name, str):
        raise InternalError(F"vector name must be str! Found type {type(name)} for {name}")
    if not name:
        raise InternalError("vector name can't be empty! ")

    if not name.startswith(enums.TensorField.vector_prefix):
        raise InternalError(
            f"Names of vectors must begin "
            f"with the vector prefix ({enums.TensorField.vector_prefix})! \n"
            f"The name of the vector that raised the error: {name}")
    without_prefix = name.replace(enums.TensorField.vector_prefix, '', 1)
    if not without_prefix:
        raise InternalError(
            f"Vector name without prefix cannot be empty. "
            f"The name of the vector that raised the error: {name}"
        )
    if without_prefix in enums.TensorField.__dict__.values():
        raise InternalError(
            f"Vector name without vector prefix can't be a protected name."
            f"The name of the vector that raised the error: {name}"
        )
    if without_prefix == '_id':
        raise InternalError(
            f"Vector name without vector prefix can't be a protected name."
            f"The name of the vector that raised the error: {name}"
        )
    return name


def validate_searchable_vector_props(existing_vector_properties: Container[str],
                                     subset_vector_properties: Iterable[str]) -> Iterable[str]:
    """Validates that the a subset of vector properties is indeed a subset.

    Args:
        existing_vector_properties: assumes that each name begins with the vector prefix
        subset_vector_properties: assumes that each name begins with the vector prefix

    Returns:
        subset_vector_properties if validation has passed
    Raises
        S2Search error in case where subset_vector_properties isn't a subset of
        existing_vector_properties

    """
    for subset_vec in subset_vector_properties:
        if subset_vec not in existing_vector_properties:
            raise MarqoError(f"Searchable attribute '{subset_vec.replace(TensorField.vector_prefix, '')}' "
                             f"not found in index.")
    return subset_vector_properties


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
        raise InvalidDocumentIdError("Document ID must can't be empty")
    return _id


def validate_index_name(name: str) -> str:
    """Validates the index name.

    Args:
        name: the name of the index

    Returns:
        name, if no errors have been raised.

    Raises
        InvalidIndexNameError
    """
    if name in constants.INDEX_NAMES_TO_IGNORE:
        raise InvalidIndexNameError(
            f"Index name `{name}` conflicts with a protected name. "
            f"Please chose a different name for your index.")
    if any([name.startswith(protected_prefix) for protected_prefix in constants.INDEX_NAME_PREFIXES_TO_IGNORE]):
        raise InvalidIndexNameError(
            f"Index name `{name}` starts with a protected prefix. "
            f"Please chose a different name for your index.")
    return name
