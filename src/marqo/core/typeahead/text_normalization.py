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


def generate_suffixes(text: str) -> List[str]:
    """
    Generate all suffixes of text for prefix matching.
    
    Args:
        text: Input text to generate suffixes for
        
    Returns:
        List of all suffixes including the full text
    """
    if not text:
        return []

    # Generate all suffixes
    suffixes = []
    for i in range(len(text)):
        suffix = text[i:]
        if suffix.strip():  # Only add non-empty suffixes
            suffixes.append(suffix.strip())

    return suffixes
