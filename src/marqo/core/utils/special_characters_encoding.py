import re

def custom_encode(s: str) -> str:
    return re.sub(r'(")', lambda m: r'\u0022', s)

def decode_key(encoded_key: str) -> str:
    return re.sub(r'\\u0022', '"', encoded_key)