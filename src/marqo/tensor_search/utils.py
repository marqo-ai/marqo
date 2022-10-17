import typing
from typing import List
import functools
import json
import torch
from marqo import errors
from marqo.tensor_search import enums
from typing import List, Optional, Union, Callable, Iterable, Sequence, Dict
import copy
import datetime


def dicts_to_jsonl(dicts: List[dict]) -> str:
    """Turns a list of dicts into a JSONL string"""
    return functools.reduce(
        lambda x, y: "{}\n{}".format(x, json.dumps(y)),
        dicts, ""
    ) + "\n"


def generate_vector_name(field_name: str) -> str:
    """Generates the name of the vector based on the field name"""
    return F"{enums.TensorField.vector_prefix}{field_name}"


def truncate_dict_vectors(doc: Union[dict, List], new_length: int = 5) -> Union[List, Dict]:
    """Creates a readable version of a dict by truncating identified vectors
    Looks for field names that contains the keyword "vector"
    """
    copied = copy.deepcopy(doc)

    if isinstance(doc, list):
        return [truncate_dict_vectors(d, new_length=new_length)
                if isinstance(d, list) or isinstance(d, dict)
                else copy.deepcopy(d)
                for d in doc]

    for k, v in list(copied.items()):
        if "vector" in k.lower() and isinstance(v, Sequence):
            copied[k] = v[:new_length]
        elif isinstance(v, dict) or isinstance(v, list):
            copied[k] = truncate_dict_vectors(v, new_length=new_length)

    return copied


def create_duration_string(timedelta):
    """Creates a duration string suitable that can be returned in the AP

    Args:
        timedelta (datetime.timedelta): time delta, or duration.

    Returns:

    """
    return f"PT{timedelta.total_seconds()}S"


def format_timestamp(timestamp: datetime.datetime):
    """Creates a timestring string suitable for return in the API

    Assumes timestamp is UTC offset 0
    """
    return f"{timestamp.isoformat()}Z"


def construct_authorized_url(url_base: str, username: str, password: str) -> str:
    """
    Args:
        url_base:
        username:
        password:

    Returns:

    """
    http_sep = "://"
    if http_sep not in url_base:
        raise errors.MarqoError(f"Could not parse url: {url_base}")
    url_split = url_base.split(http_sep)
    if len(url_split) != 2:
        raise errors.MarqoError(f"Could not parse url: {url_base}")
    http_part, domain_part = url_split
    return f"{http_part}{http_sep}{username}:{password}@{domain_part}"


def contextualise_filter(filter_string: str, simple_properties: typing.Iterable) -> str:
    """adds the chunk prefix to the start of properties found in simple string

    This allows for filtering within chunks.

    Args:
        filter_string:
        simple_properties: simple properties of an index (such as text or floats
            and bools)

    Returns:
        a string where the properties are referenced as children of a chunk.
    """
    contextualised_filter = filter_string
    for field in simple_properties:
        contextualised_filter = contextualised_filter.replace(f'{field}:', f'{enums.TensorField.chunks}.{field}:')
    return contextualised_filter


def check_device_is_available(device: str) -> bool:
    """Checks if a device is available on the machine

    Args:
        device: assumes device is a valid device strings (e.g: 'cpu' or
            'cuda:1')

    Returns:
        True, IFF it is available

    Raises:
        MarqoError if device is determined to be invalid
    """
    lowered = device.lower()
    if lowered == "cpu":
        return True

    split = lowered.split(":")
    if split[0] != "cuda":
        raise errors.MarqoError(f"Invalid device prefix! {device}. Valid prefixes: 'cpu' and 'cuda'")

    if not torch.cuda.is_available():
        return False

    if len(split) < 2:
        return True

    if int(split[1]) < 0:
        raise errors.MarqoError(f"Invalid cuda device number! {device}. It must not be negative")

    if int(split[1]) < torch.cuda.device_count():
        return True
    else:
        return False


def merge_dicts(base: dict, preferences: dict) -> dict:
    """Merges two dicts together. Fields in the base dict are overwritten by
    the preferences dict
    """
    merged_dicts = copy.deepcopy(base)

    def merge(merged: dict, prefs: dict) -> dict:
        for key in prefs:
            if not isinstance(prefs[key], dict):
                merged[key] = prefs[key]
            else:
                if key in merged:
                    merged[key] = merge(merged[key], prefs[key])
                else:
                    merged[key] = prefs[key]
        return merged

    return merge(merged=merged_dicts, prefs=preferences)
