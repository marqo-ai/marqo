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

multimodal_combination_mappings_schema = {
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
          # TODO: add child_field in constants.ALLOWED_MULTIMODAL_FIELD_TYPES:
          # TODO: add weights are float or int only
        },
      },

    }
  },
  "required": [
    "type",
    "weights"
  ], "additionalProperties": False
}


custom_vector_mappings_schema = {
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "object",
  "properties": {
    "type": {
      "type": "string",
      "enum": [MappingsObjectType.custom_vector]
    }
  },
  "required": ["type"],
  "additionalProperties": False
}