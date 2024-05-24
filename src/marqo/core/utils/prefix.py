
from marqo.s2_inference.errors import InvalidModelPropertiesError
from marqo.core.models.marqo_index import *
from marqo.s2_inference.s2_inference import get_model_properties_from_registry

class DeterminePrefixContentType(Enum):
    TextChunk = "text_chunk_prefix"
    TextQuery = "text_query_prefix"

def determine_text_prefix(request_level_prefix: str, marqo_index: MarqoIndex, prefix_type: DeterminePrefixContentType) -> str:
    """
    Determines the text prefix to be used for chunking text fields or search queries.
    This prefix will be added before each text chunk or query to enhance processing accuracy.
        
    Logic:
    1. Prioritize request-level prefix
    2. If not provided, use settings based on prefix_type
    3. If still not provided, use model_properties defined prefix
    4. Return "" if no prefix is found, handling is expected by the caller

    Args:
        request_level_prefix (str): The prefix provided in the request
        index_settings (IndexSettings): The settings object containing prefix information
        prefix_type (str): Either "text_query_prefix" or "text_chunk_prefix"

    Returns:
        str: The determined prefix, or None if no prefix is found
    """
    if request_level_prefix is not None:
        return request_level_prefix
    

    # Check for the presence of the textChunkPrefix or textQueryPrefix in the MarqoIndex object.
    if prefix_type == DeterminePrefixContentType.TextQuery and marqo_index.override_text_query_prefix is not None:
        return marqo_index.override_text_query_prefix
    elif prefix_type == DeterminePrefixContentType.TextChunk and marqo_index.override_text_chunk_prefix is not None:
        return marqo_index.override_text_chunk_prefix

    # Fallback to model_properties defined prefix
    try:
        model_properties = marqo_index.model.properties if marqo_index.model is not None else None
        if model_properties is None and marqo_index.model is not None:
            model_properties = get_model_properties_from_registry(marqo_index.model.name)

        if model_properties is not None:
            default_prefix = model_properties.get(prefix_type.value)
            return default_prefix if default_prefix is not None else ""
        else:
            raise ValueError(f"Model properties not found for model: {marqo_index.model.name}")
    except InvalidModelPropertiesError as e:
        raise e
    
    # If no prefix is found, return empty string ""
    return ""



