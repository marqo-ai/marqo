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
    """Recursively constructs the filter used to narrow the search down to specific searchable attributes"""
    if searchable_attribs is None or len(searchable_attribs) == 0:
        return ""

    vector_prop_count = len(searchable_attribs)

    # brackets surround field name, in case it contains a space:
    sanitised_attr_name = f"({sanitise_lucene_special_chars(searchable_attribs.pop())})"
    
    # base case
    if vector_prop_count == 1:
        return f"{enums.TensorField.chunks}.{enums.TensorField.field_name}:{sanitised_attr_name}"
    else:
        return (
            f"{enums.TensorField.chunks}.{enums.TensorField.field_name}:{sanitised_attr_name}"
            f" OR {build_searchable_attributes_filter(searchable_attribs=searchable_attribs)}")


def sanitise_lucene_special_chars(to_be_sanitised: str) -> str:
    """Santitises Lucene's special chars in a string.

    We shouldn't apply this to the user's filter string, as they can choose to escape
    Lucene's special chars themselves.

    This should be used to sanitise a filter string constructed for users behind the
    scenes (such as for searchable attributes).

    See here for more info:
    https://lucene.apache.org/core/6_0_0/queryparser/org/apache/lucene/queryparser/classic/package-summary.html#Escaping_Special_Characters

    """
    
    # always escape backslashes before the other special chars
    to_be_sanitised = to_be_sanitised.replace("\\", "\\\\")

    # this prevents us from double-escaping backslashes.
    non_backslash_chars = constants.LUCENE_SPECIAL_CHARS.union(constants.NON_OFFICIAL_LUCENE_SPECIAL_CHARS) - {'\\'}

    for char in non_backslash_chars:
        to_be_sanitised = to_be_sanitised.replace(char, f'\\{char}')
    return to_be_sanitised


def contextualise_user_filter(filter_string: Optional[str], simple_properties: typing.Iterable) -> str:
    """adds the chunk prefix to the start of properties found in simple string (filter_string)
    This allows for filtering within chunks.

    Because this is a user-defined filter, if they want to filter on field names that contain
    special characters, we expect them to escape the special characters themselves.

    In order to search chunks we need to append the chunk prefix to the start of the field name.
    This will only work if they escape the special characters in the field names themselves in
    the exact same way that we do.

    Args:
        filter_string: the user defined filter string
        simple_properties: simple properties of an index (such as text or floats
            and bools)

    Returns:
        a string where the properties are referenced as children of a chunk.
    """
    if filter_string is None:
        return ''
    if simple_properties is None:
        return filter_string
    
    contextualised_filter = filter_string

    for field in simple_properties:
        escaped_field_name = sanitise_lucene_special_chars(field)
        if escaped_field_name in filter_string:
            # we want to replace the field name that directly corresponds to the simple property,
            # not any other field names that contain the simple property as a substring.

            # case 0: field name is at the start of the filter string
            # it must be followed by a colon, otherwise it is a substring of another field name
            # edge case example: "field_a_excess_chars:a, escaped_field_name=field_a"
            if filter_string.startswith(escaped_field_name + ":"):
                # add the chunk prefix ONLY to the start of the field name
                contextualised_filter = f'{enums.TensorField.chunks}.{contextualised_filter}'
            
            # next we check every occurence of field name NOT at the start of the filter string
            # note: we do this even if it was also at the start
            # case 1: field name is after a space
            # e.g.: "field_a:foo AND field_b:bar, escaped_field_name=field_b"
            contextualised_filter = contextualised_filter.replace(
                f' {escaped_field_name}:', f' {enums.TensorField.chunks}.{escaped_field_name}:')

            # case 2: field name is directly after an opening parenthesis
            contextualised_filter = contextualised_filter.replace(
                f'({escaped_field_name}:', f'({enums.TensorField.chunks}.{escaped_field_name}:')
    return contextualised_filter