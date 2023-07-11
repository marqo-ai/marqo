import os
import typing
from timeit import default_timer as timer
from marqo import errors
from marqo.tensor_search import enums, configs, constants
from typing import (
    List, Optional, Union, Callable, Iterable, Sequence, Dict, Tuple
)
from marqo.marqo_logging import logger
import copy
from marqo.tensor_search.enums import EnvVars

def build_tensor_search_filter(
        filter_string: str, simple_properties: dict,
        searchable_attribs: Sequence):
    """Builds a Lucene-DSL filter string for OpenSearch, that combines the user's filter string
    with searchable_attributes

    """
    if searchable_attribs is not None:
        copied_searchable_attribs = copy.deepcopy(searchable_attribs)
        searchable_attribs_filter = build_searchable_attributes_filter(
            searchable_attribs=copied_searchable_attribs)
    else:
        searchable_attribs_filter = ""

    contextualised_user_filter = contextualise_user_filter(
        filter_string=filter_string, simple_properties=simple_properties)

    if contextualised_user_filter and searchable_attribs_filter:
        return f"({searchable_attribs_filter}) AND ({contextualised_user_filter})"
    else:
        return f"{searchable_attribs_filter}{contextualised_user_filter}"


def build_searchable_attributes_filter(searchable_attribs: Sequence) -> str:
    """Constructs the filter used to narrow the search down to specific searchable attributes"""
    if len(searchable_attribs) == 0:
        return ""

    vector_prop_count = len(searchable_attribs)

    # brackets surround field name, in case it contains a space:
    sanitised_attr_name = f"({sanitise_lucene_special_chars(searchable_attribs.pop())})"

    if vector_prop_count == 1:
        return f"{enums.TensorField.chunks}.{enums.TensorField.field_name}:{sanitised_attr_name}"
    else:
        return (
            f"{enums.TensorField.chunks}.{enums.TensorField.field_name}:{sanitised_attr_name}"
            f" OR {build_searchable_attributes_filter(searchable_attribs=searchable_attribs)}")

def sanitise_lucene_special_chars(user_str: str) -> str:
    """Santitises Lucene's special chars.

    See here for more info:
    https://lucene.apache.org/core/6_0_0/queryparser/org/apache/lucene/queryparser/classic/package-summary.html#Escaping_Special_Characters

    """
    for char in constants.LUCENE_SPECIAL_CHARS:
        user_str = user_str.replace(char, f'\\{char}')
    return user_str


def contextualise_user_filter(filter_string: Optional[str], simple_properties: typing.Iterable) -> str:
    """adds the chunk prefix to the start of properties found in simple string
    This allows for filtering within chunks.

    Args:
        filter_string:
        simple_properties: simple properties of an index (such as text or floats
            and bools)

    Returns:
        a string where the properties are referenced as children of a chunk.
    """
    if filter_string is None:
        return ''
    contextualised_filter = filter_string
    for field in simple_properties:
        if ' ' in field:
            field_with_escaped_space = field.replace(' ', r'\ ') # monitor this: fixed the invalid escape sequence (Deprecation warning).
            contextualised_filter = contextualised_filter.replace(f'{field_with_escaped_space}:', f'{enums.TensorField.chunks}.{field_with_escaped_space}:')
        else:
            contextualised_filter = contextualised_filter.replace(f'{field}:', f'{enums.TensorField.chunks}.{field}:')
    return contextualised_filter