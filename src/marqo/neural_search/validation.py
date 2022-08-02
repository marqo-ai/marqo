from marqo.neural_search import enums
from typing import Iterable, Container
from marqo.errors import MarqoError
from marqo.neural_search.enums import NeuralField
from marqo.neural_search import constants

def validate_field_name(field_name) -> str:
    """TODO:
        - length (remember the vector name will have the vector_prefix added to the front of field_name)
        - consider blanket "no double names starting with double underscore..."
    Args:
        field_name:

    returns field_name, if all validations pass

    Raises:
        MarqoError
    """
    if not isinstance(field_name, str):
        raise MarqoError(F"field name must be str! Found type {type(field_name)} for {field_name}")
    if not field_name:
        raise MarqoError("field name can't be empty! ")
    if field_name.startswith(enums.NeuralField.vector_prefix):
        raise MarqoError(F"can't start field name with protected prefix {enums.NeuralField.vector_prefix}."
                            F" Error raised for field name: {field_name}")
    if field_name.startswith(enums.NeuralField.chunks):
        raise MarqoError(F"can't name field with protected field name {enums.NeuralField.chunks}."
                            F" Error raised for field name: {field_name}")
    char_validation = [(c, c not in constants.ILLEGAL_CUSTOMER_FIELD_NAME_CHARS)
                        for c in field_name]
    char_validation_failures = [c for c in char_validation if not c[1]]
    if char_validation_failures:
        raise MarqoError(F"Illegal character '{char_validation_failures[0][0]}' "
                            F"detected in field name {field_name}")
    if field_name not in enums.NeuralField.__dict__.values():
        return field_name
    else:
        raise MarqoError(f"field name can't be a protected field. Please rename this field: {field_name}")


def validate_doc(doc: dict) -> dict:
    """
    Args:
        doc: a document indexed by the client

    Raises an MarqoError

    Returns
        doc if all validations pass
    """
    if len(doc) <= 0:
        raise MarqoError("Can't index empty dict!")
    return doc


def validate_vector_name(name: str):
    """Checks that the vector name is valid.
    It should have the form __vector_{customer field name}
    """
    if not isinstance(name, str):
        raise MarqoError(F"vector name must be str! Found type {type(name)} for {name}")
    if not name:
        raise MarqoError("vector name can't be empty! ")

    if not name.startswith(enums.NeuralField.vector_prefix):
        raise MarqoError(
            f"Names of vectors must begin "
            f"with the vector prefix ({enums.NeuralField.vector_prefix})! \n"
            f"The name of the vector that raised the error: {name}")
    without_prefix = name.replace(enums.NeuralField.vector_prefix, '', 1)
    if not without_prefix:
        raise MarqoError(
            f"Vector name without prefix cannot be empty. "
            f"The name of the vector that raised the error: {name}"
        )
    if without_prefix in enums.NeuralField.__dict__.values():
        raise MarqoError(
            f"Vector name without vector prefix can't be a protected name."
            f"The name of the vector that raised the error: {name}"
        )
    if without_prefix == '_id':
        raise MarqoError(
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
            raise MarqoError(f"Searchable attribute '{subset_vec.replace(NeuralField.vector_prefix, '')}' "
                                f"not found in index.")
    return subset_vector_properties
