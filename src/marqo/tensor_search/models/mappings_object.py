from marqo.tensor_search.enums import MappingsObjectType
from marqo.tensor_search.constants import MARQO_OBJECT_TYPES


mappings_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "patternProperties": {
        "^.*$": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": list(MARQO_OBJECT_TYPES)
                },
            },
            "required": [
                "type",
            ]
        }
    },
}

multimodal_combination_schema = {
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "object",
  "properties": {
    "type": {
      "type": "string",
      "enum": [MappingsObjectType.multimodal_combination]
    },
    "weights": {
      "type": "object",
      "patternProperties": {
        "^.*$": {
          "type": "number"
        },
      },

    }
  },
  "required": [
    "type",
    "weights"
  ], "additionalProperties": False
}


custom_vector_schema = {
    # ???
}