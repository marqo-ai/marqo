import re

def custom_encode(s: str) -> str:
    return re.sub(r'(")|\\u0022', lambda m: r'\u0022' if m.group(1) else r'\\u0022', s)

def decode_key(encoded_key: str) -> str:
    return re.sub(r'\\u0022', '"', encoded_key)