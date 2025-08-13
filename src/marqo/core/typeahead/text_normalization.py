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
    Generate all suffixes of normalized text for prefix matching.
    
    Args:
        text: Input text to generate suffixes for
        
    Returns:
        List of all suffixes including the full text
    """
    if not text:
        return []
    
    normalized = normalize_text(text)
    if not normalized:
        return []
    
    # Generate all suffixes
    suffixes = []
    for i in range(len(normalized)):
        suffix = normalized[i:]
        if suffix.strip():  # Only add non-empty suffixes
            suffixes.append(suffix.strip())
    
    return suffixes


def calculate_edit_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance between the strings
    """
    if len(s1) < len(s2):
        return calculate_edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]