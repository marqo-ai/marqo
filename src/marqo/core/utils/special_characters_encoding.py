import urllib.parse


def custom_encode(s: str) -> str:
    encoded_string = ''.join(c if c.isascii() and c not in '"\\/' else urllib.parse.quote(c, safe='') for c in s)
    return encoded_string

def decode_key(encoded_key: str) -> str:
    return urllib.parse.unquote(encoded_key)
