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

from fastapi import HTTPException

from marqo import logging
from marqo.api import exceptions, configs
from marqo.tensor_search import enums
from marqo.tensor_search.enums import EnvVars
from marqo.core.constants import CHARACTERS_TO_BE_ESCAPED_IN_VESPA


logger = logging.get_logger(__name__)

def dicts_to_jsonl(dicts: List[dict]) -> str:
    """Turns a list of dicts into a JSONL string"""
    return functools.reduce(
        lambda x, y: "{}\n{}".format(x, json.dumps(y)),
        dicts, ""
    ) + "\n"


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
        raise exceptions.MarqoError(f"Could not parse url: {url_base}")
    url_split = url_base.split(http_sep)
    if len(url_split) != 2:
        raise exceptions.MarqoError(f"Could not parse url: {url_base}")
    http_part, domain_part = url_split
    return f"{http_part}{http_sep}{username}:{password}@{domain_part}"


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
        raise exceptions.ConfigurationError(value_error_msg)
    return as_int


def parse_lexical_query(text: str) -> Tuple[List[str], List[str]]:
    r"""
    Find required terms enclosed within double quotes.

    All other terms go into blob, split by whitespace. blob starts as a string then splits into
    a list by space.

    Backslash will be used to escape " or \, but if it is not followed by " or \,
    it should be IGNORED. This is to prevent hanging backslashes accidentally escaping query quotes or inserting bad
    characters into query (for example \a causes a Vespa 400).

    Syntax:
        Required strings must be enclosed by quotes. These quotes must be enclosed by spaces or the start
        or end of the text. Quotes always come in pairs.

        Bad syntax rules:
        - If 1 or both of the quotes in a pair has bad syntax (e.g. no space before opening quote or after closing
          quote), both will be treated as whitespace.
        - Incomplete quotes will be treated as whitespace.
    Notes:
        - Correct double quote can be either opening, closing, or escaped.
        - Escaped double quotes are interpreted literally.
        - Escaped backslashes are interpreted literally

    PYTHON NOTE:
    Users using python need to escape the backslash itself. (Single \ get ignored) -> q='dwayne \\"the rock\\" johnson'
    Unneeded if using CLI or raw string.

    Return:
        2-tuple of <required terms> (for "must" clause) <optional terms> (for "should" clause)
    """

    required_terms = []
    blob = ""
    opening_quote_idx = None
    current_quote_pair_is_faulty = False
    escape = False

    if not isinstance(text, str):
        raise TypeError("parse_lexical_query must have string as input")

    for i in range(len(text)):
        # Every character immediately after a \\ should be read literally
        if escape:
            escape = False
            blob += text[i]
        elif text[i] == "\\":
            escape = True
            # Stray backslashes should be ignored (not followed by special char, or is last char)
            if not (i == len(text) - 1 or text[i + 1] not in CHARACTERS_TO_BE_ESCAPED_IN_VESPA):
                blob += text[i]
        elif text[i] == '"':
            # OPENING QUOTE
            if (opening_quote_idx is None):
                opening_quote_idx = i
                blob_opening_quote_idx = len(blob) # Opening quote index in blob is different from text

                # Bad syntax opening quote: flag it, replace quote with whitespace
                if not (i == 0 or text[i - 1] == " "):
                    current_quote_pair_is_faulty = True
                    blob += " "
                else:
                    # Good syntax opening quote: add to blob
                    blob += text[i]
            # CLOSING QUOTE
            else:
                # Good syntax closing: must have space on the right (or is last character) while opening exists.
                if (i == len(text) - 1 or text[i + 1] == " ") and not current_quote_pair_is_faulty:
                    # Add this quote to the blob
                    blob += text[i]
                    # Add everything in between the quotes as a required term
                    new_required_term = text[opening_quote_idx + 1:i]
                    if new_required_term:                           # Do not add empty strings as required terms
                        required_terms.append(new_required_term)

                    # Remove this required term from the blob
                    blob = blob[:-(len(new_required_term) + 2)]

                else:
                    # Bad syntax closing: treat this and opening quote as whitespace
                    blob = blob[:blob_opening_quote_idx] + " " + \
                                     blob[blob_opening_quote_idx + 1:] + " "

                # Clean up flags
                opening_quote_idx = None
                current_quote_pair_is_faulty = False
        else:
            # If not a special character, add to blob
            blob += text[i]

    # Unpaired quote will be turned to whitespace
    if opening_quote_idx is not None:
        blob = blob[:blob_opening_quote_idx] + " " + blob[blob_opening_quote_idx + 1:]

    # Remove double/leading white spaces
    optional_terms = blob.split()

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


def is_tensor_field(field: str,
                    tensor_fields: List[str]
                    ) -> bool:
    """Determine whether a field is a tensor field or not for add_documents calls."""
    if not tensor_fields:
        return False
    else:
        return field in tensor_fields


def check_is_zero_vector(vector: List[float]) -> bool:
    """Check if a vector is all zero. We assume the input to this function is of valid type, List[Float]"""
    return all([x == 0 for x in vector])


def extract_multimodal_mappings(mappings: Dict) -> Dict:
    """Extract multimodal mappings from mappings dict"""
    return {k: v for k, v in mappings.items() if v["type"] == "multimodal_combination"}


def extract_multimodal_content(doc: dict, mapping: Dict) -> Dict:
    """Extract multimodal content from doc based on multimodal mapping"""
    multimodal_content = {}
    for field_name, weight in mapping["weights"].items():
        if field_name in doc:
            multimodal_content[field_name] = doc[field_name]
    return multimodal_content


def enable_batch_apis():
    def decorator_function(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if read_env_vars_and_defaults(EnvVars.MARQO_ENABLE_BATCH_APIS).lower() != 'true':
                raise HTTPException(status_code=403,
                                    detail="This API endpoint is disabled. Please set MARQO_ENABLE_BATCH_API to true to enable it.")
            return func(*args, **kwargs)

        return wrapper

    return decorator_function


def enable_upgrade_api():
    def decorator_function(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if read_env_vars_and_defaults(EnvVars.MARQO_ENABLE_UPGRADE_API).lower() != 'true':
                raise HTTPException(status_code=403,
                                    detail="This API endpoint is disabled. Please set MARQO_ENABLE_UPGRADE_API to true to enable it.")
            return func(*args, **kwargs)

        return wrapper

    return decorator_function


def enable_debug_apis():
    def decorator_function(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if read_env_vars_and_defaults(EnvVars.MARQO_ENABLE_DEBUG_API).lower() != 'true':
                raise HTTPException(status_code=403,
                                    detail="This API endpoint is disabled. Please set MARQO_ENABLE_DEBUG_API to true to enable it.")
            return func(*args, **kwargs)

        return wrapper

    return decorator_function


def enable_ops_api():
    def decorator_function(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if read_env_vars_and_defaults(EnvVars.MARQO_ENABLE_OPS_API).lower() != 'true':
                raise HTTPException(status_code=403,
                                    detail="This API endpoint is disabled. Please set MARQO_ENABLE_OPS_API to true to enable it.")
            return func(*args, **kwargs)

        return wrapper

    return decorator_function
