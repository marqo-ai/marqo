from typing import Any, Dict, List, Optional, Union
from types import FunctionType

from functools import partial
from more_itertools import windowed

from nltk.tokenize import sent_tokenize, word_tokenize
import nltk


def _splitting_functions(split_by: str, language: str='english') -> FunctionType:
    """_summary_
    selects a text splitting function based on the method provided by 'split_by'
    Args:
        split_by (str): method to split the text by, 'character', 'word', 'sentence', 'passage'
                        if not one of those allows for custom characters to split on
        language (str, optional): _description_. Defaults to 'english'.

    Raises:
        TypeError: _description_

    Returns:
        _type_: function for splitting text based on the method provided 
    """
    if not isinstance(split_by, str):
        raise TypeError(f"expected str received {type(split_by)}")

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    MAPPING = {
        'character':list,
        'word': partial(word_tokenize, language=language),
        'sentence':partial(sent_tokenize, language=language),
        'passage':lambda x:x.split("\n\n")
    }

    if split_by in MAPPING:
        return MAPPING[split_by]
    
    # can also have a custom split but leave that for now
    raise KeyError(f"unexpected split_by type of {split_by}")

def _reconstruct_single_list(segmented_text: List[str], seperator: str = " ") -> str:
    """_summary_

    Args:
        segmented_text (List[str]): List of strings that were segmented 
        seperator (str, optional): what to use as a seperator between tokens when re-joining into text. Defaults to " ".

    Returns:
        str: string of text that was re-joined
    """
    return seperator.join([t for t in segmented_text if t is not None])

def _reconstruct_multi_list(segmented_text_list: List[List[str]], seperator: str = " ") -> List[str]:
    """_summary_

    Args:
        segmented_text_list (List[List[str]]): _description_
        seperator (str, optional): _description_. Defaults to " ".

    Returns:
        List[str]: _description_
    """

    results = []
    for seg in segmented_text_list:
        txt = _reconstruct_single_list(seg, seperator)
        if len(txt) > 0:
            results.append(txt)
    
    return results

def check_make_string_valid(text: str, coerce: bool = True) -> str:
    """ does some simple validation and coercsion for empty strings

    Args:
        text (str): text of type str
        coerce (bool, optional): if an empty or None string is passed, return an empty string?. Defaults to True.

    Raises:
        TypeError: _description_

    Returns:
        _type_: something that is a string
    """
    empty_string = " "

    if text in [[], None, '', "", empty_string] and coerce:
        return empty_string

    if set(text) == set(" "):
        return empty_string

    if not isinstance(text, str):
        raise TypeError(f"text had type {type(text)} but expected str")

    return text

def split_text(text: str, split_by: str = 'sentence', split_length: int = 2, split_overlap: int = 1, 
               language: str = 'english', custom_seperator: str = None) -> List[str]:
    """ splits a single piece of text into smaller sub-texts based on splitting method (split_by).
        for example, the text can can be split at the character, word, sentence or passage level.
        optionally it can be split with a custom splitting string

    Args:
        text (str): _description_
        split_by (str, optional): _description_. Defaults to 'character'.
        split_length (int, optional): _description_. Defaults to 128.
        split_overlap (int, optional): _description_. Defaults to 16.
        language (str, optional): _description_. Defaults to 'english'.
        seperator (str, optional): _description_. Defaults to " ".

    Raises:
        TypeError: _description_

    Returns:
        List[str]: _description_
    """

    if split_length == 0:
        raise ValueError("split length must be > 0")

    # simple validation and correction    
    text = check_make_string_valid(text, coerce=True)

    # don't split if it is not worth splitting
    if len(text) <= 1:
        return [text]

    # we need to treat character splitting differently
    if custom_seperator is None:
        seperator = '' if split_by == 'character' else ' '
    else: 
        seperator = custom_seperator

    # determine how we want to split
    _func = _splitting_functions(split_by, language=language)

    # do the splitting
    split_text = _func(text)
    
    # concatenate individual elements based on split_length & split_stride
    segments = list(windowed(split_text, n=split_length, step=split_length - split_overlap))

    # reconstruct the segments. there is potential for a lossy process here as we
    # assume a uniform seperator when reconstructing the sentences
    text_splits = _reconstruct_multi_list(segments, seperator)

    return text_splits

