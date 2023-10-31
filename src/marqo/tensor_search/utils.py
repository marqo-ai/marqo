import copy
import datetime
import functools
import json
import os
import pathlib
from timeit import default_timer as timer
from typing import (
    List, Optional, Union, Sequence, Dict, Tuple
)

import torch

from marqo import errors
from marqo.marqo_logging import logger
from marqo.tensor_search import enums, configs
from marqo.tensor_search.enums import EnvVars


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
            if prefs[key] is None:
                continue
            if not isinstance(prefs[key], dict):
                merged[key] = prefs[key]
            else:
                if key in merged:
                    merged[key] = merge(merged[key], prefs[key])
                else:
                    merged[key] = prefs[key]
        return merged

    return merge(merged=merged_dicts, prefs=preferences)


def read_env_vars_and_defaults(var: str) -> Optional[str]:
    """Attempts to read an environment variable.
    If none is found, it will attempt to retrieve it from
    configs.default_env_vars(). If still unsuccessful, None is returned.
    If it's an empty string, None is returned.
    """

    def none_if_empty(value: Optional[str]) -> Optional[str]:
        """Returns None if value is an empty string"""
        if value is not None and len(value) == 0:
            return None
        else:
            return value

    try:
        return none_if_empty(os.environ[var])
    except KeyError:
        try:
            default_val = configs.default_env_vars()[var]
            if isinstance(default_val, str):
                return none_if_empty(default_val)
            else:
                return default_val
        except KeyError:
            return None


def read_env_vars_and_defaults_ints(var: str) -> Optional[int]:
    """Gets env var from read_env_vars_and_defaults() and attempts to coerce it to an int

    Returns
        the coerced int value, or None if the key is not found.
    """
    str_val = read_env_vars_and_defaults(var)

    if str_val is None:
        return None

    validation_error_msg = (
        f"Could not properly read env var `{var}`. `{var}` must be able to be parsed as an int."
    )
    try:
        as_int = int(str_val)
    except (ValueError, TypeError) as e:
        value_error_msg = f"`{validation_error_msg} Current value: `{str_val}`. Reason: {e}"
        logger.error(value_error_msg)
        raise errors.ConfigurationError(value_error_msg)
    return as_int


def parse_lexical_query(text: str) -> Tuple[List[str], List[str]]:
    """
    Find required terms enclosed within double quotes.

    All other terms go into optional_terms, split by whitespace.

    Syntax:
        Required strings must be enclosed by quotes. These quotes must be enclosed by spaces or the start
        or end of the text

    Notes:
        Double quote can be either opening, closing, or escaped.
        Escaped double quotes are interpreted literally.
        If any double quotes exist that are neither opening, closing, nor escaped,
        interpret entire string literally instead.

    Users need to escape the backslash itself. (Single \ get ignored) -> q='dwayne \\"the rock\\" johnson'

    Return:
        2-tuple of <required terms> (for "must" clause) <optional terms> (for "should" clause)
    """
    required_terms = []
    optional_terms = ""
    opening_quote_idx = None

    if not isinstance(text, str):
        raise TypeError("parse_lexical_query must have string as input")

    for i in range(len(text)):
        # Add all characters to blob initially
        optional_terms += text[i]

        if text[i] == '"':
            # Check if ESCAPED
            if i > 0 and text[i - 1] == '\\':
                # Read quote literally. Backslash should be ignored (both blob and required)
                pass

            # Check if CLOSING QUOTE
            # Closing " must have space on the right (or is last character) while opening exists.
            elif (opening_quote_idx is not None) and (i == len(text) - 1 or text[i + 1] == " "):
                # Add everything in between the quotes as a required term
                new_required_term = text[opening_quote_idx + 1:i]
                required_terms.append(new_required_term)

                # Remove this required term from the optional blob
                optional_terms = optional_terms[:-(len(new_required_term) + 2)]
                opening_quote_idx = None

            # Check if OPENING QUOTE
            # Opening " must have space on the left (or is first character).
            elif i == 0 or text[i - 1] == " ":
                opening_quote_idx = i

            # None of the above: Syntax error. Interpret text literally instead.
            else:
                return ([], text)

    if opening_quote_idx is not None:
        # string parsing finished with a quote still open: syntax error.
        return ([], text)

    # Remove double/leading white spaces
    optional_terms = optional_terms.split()

    # Remove escape character. `\"` becomes just `"`
    required_terms = [term.replace('\\"', '"') for term in required_terms]
    optional_terms = [term.replace('\\"', '"') for term in optional_terms]

    return required_terms, optional_terms


def get_marqo_root_from_env() -> str:
    """Returns absolute path to Marqo root, first checking the env var.

    If it isn't found, it creates the env var and returns it.

    Returns:
        str that doesn't end in a forward slash.
        for example: "/Users/CoolUser/marqo/src/marqo"
    """
    try:
        marqo_root_path = os.environ[enums.EnvVars.MARQO_ROOT_PATH]
        if marqo_root_path:
            return marqo_root_path
    except KeyError:
        pass
    mq_root = _get_marqo_root()
    os.environ[enums.EnvVars.MARQO_ROOT_PATH] = mq_root
    return mq_root


def _get_marqo_root() -> str:
    """returns absolute path to Marqo root

    Searches for the Marqo by examining its own file path.

    Returns:
        str that doesn't end in a forwad in forward slash.
        for example: "/Users/CoolUser/marqo/src/marqo"
    """
    # tensor_search/ is the parent dir of this file
    tensor_search_dir = pathlib.Path(__file__).parent
    # marqo is the parent of the tensor_search_dir
    marqo_base_dir = tensor_search_dir.parent.resolve()
    return str(marqo_base_dir)


def add_timing(f, key: str = "processingTimeMs"):
    """ Function decorator to add function timing to response payload.

    Decorator for functions that adds the processing time to the return Dict (NOTE: must return value of function must
    be a dictionary). `key` param denotes what the processing time will be stored against.
    """

    @functools.wraps(f)
    def wrap(*args, **kw):
        t0 = timer()
        r = f(*args, **kw)
        time_taken = timer() - t0
        r[key] = round(time_taken * 1000)
        return r

    return wrap


def generate_batches(seq: Sequence, batch_size: int):
    """Yields batches of length k from the sequence."""
    if batch_size < 1:
        raise ValueError("Batch size must be greater than 0")

    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def get_best_available_device() -> str:
    """Get the best available device for Marqo to use and validate it."""
    device = read_env_vars_and_defaults(EnvVars.MARQO_BEST_AVAILABLE_DEVICE)
    if device is None or not check_device_is_available(device):
        raise errors.InternalError(
            f"Marqo encountered an error when loading device from environment variable `MARQO_BEST_AVAILABLE_DEVICE`. "
            f"Invalid device: {device}. Must be either 'cpu' or start with 'cuda'.")
    return device


def is_tensor_field(field: str,
                    tensor_fields: Optional[List[str]] = None,
                    non_tensor_fields: Optional[List[str]] = None
                    ) -> bool:
    """Determine whether a field is a tensor field or not for add_documents calls."""
    if tensor_fields is not None and non_tensor_fields is not None or \
            tensor_fields is None and non_tensor_fields is None:
        raise errors.InternalError("Must provide exactly one of tensor_fields or non_tensor_fields.")

    if tensor_fields is not None:
        return field in tensor_fields
    else:
        return field not in non_tensor_fields


def calculate_health_status(marqo_os_health_check_response: Optional[Dict]) -> dict:
    """Calculate the health status of Marqo based on the response API from Marqo-os ."""
    statuses = {
        "green": 0,
        "yellow": 1,
        "red": 2,
    }

    marqo_status = "green"

    if marqo_os_health_check_response is not None:
        if "status" in marqo_os_health_check_response:
            marqo_os_status = marqo_os_health_check_response['status']
        else:
            marqo_os_status = "red"
    else:
        marqo_os_status = "red"

    marqo_status = marqo_status if statuses[marqo_status] >= statuses[marqo_os_status] else marqo_os_status

    return marqo_status, marqo_os_status


def check_is_zero_vector(vector: List[float]) -> bool:
    """Check if a vector is all zero. We assume the input to this function is of valid type, List[Float]"""
    return all([x == 0 for x in vector])
