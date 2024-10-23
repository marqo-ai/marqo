import re

def custom_encode(s: str) -> str:
    """
    Encode double quotes in a string by replacing them with their Unicode escape sequence.

    This function replaces all occurrences of double quotes (") in the input string
    with the Unicode escape sequence '\u0022'.

    Args:
        s (str): The input string to be encoded.

    Returns:
        str: The encoded string with double quotes replaced by '\u0022'.
    """
    return re.sub(r'(")', lambda m: r'\u0022', s)

def decode_key(encoded_key: str) -> str:
    """
    Decode a string by replacing Unicode escape sequences for double quotes with actual double quotes.

    This function replaces all occurrences of the Unicode escape sequence '\u0022'
    in the input string with double quotes (").

    Args:
        encoded_key (str): The input string to be decoded.

    Returns:
        str: The decoded string with '\u0022' replaced by double quotes.
    """
    return re.sub(r'\\u0022', '"', encoded_key)
