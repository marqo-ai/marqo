from marqo.tensor_search.enums import MappingsObjectType
from marqo.tensor_search.constants import ALLOWED_CUSTOM_VECTOR_CONTENT_TYPES


def custom_vector_schema(index_model_dimensions: int) -> dict:
  return {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "properties": {
        "content": {
            "type": list(ALLOWED_CUSTOM_VECTOR_CONTENT_TYPES)
        },
        "vector": {
            "type": "array",
            "minItems": index_model_dimensions,
            "maxItems": index_model_dimensions,
            "items": { 
                "type": "number"
            }
        }
    },
    "required": ["vector"],     # content is optional.
    "additionalProperties": False
  }