import unicodedata
from typing import List


def normalize_text(text: str) -> str:
    """
    Normalize text by removing accents and converting to lowercase.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text with accents removed and lowercased
    """
    if not text:
        return ""

    # Normalize to NFKD form and remove accents
    normalized = unicodedata.normalize('NFKD', text)
    # Filter out combining characters (accents)
    without_accents = ''.join(c for c in normalized if not unicodedata.combining(c))
    # Convert to lowercase
    return without_accents.lower()


def generate_prefixes(text: str) -> List[str]:
    result = []
    prefix = ""
    for ch in text:
        if ch.isspace():
            prefix = ""   # reset when hitting whitespace
        else:
            prefix += ch
            result.append(prefix)
    return result
