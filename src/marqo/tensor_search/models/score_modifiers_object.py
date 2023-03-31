score_modifiers_object_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "multiply_score_by": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "properties": {
          "field_name": {
            "type": "string"
          },
          "weight": {
            "type": "number",
            "default": 1
          }
        },
        "required": [
          "field_name"
        ],
        "additionalProperties": False
      }
    },
    "add_to_score": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "properties": {
          "field_name": {
            "type": "string"
          },
          "weight": {
            "type": "number",
            "default": 1
          }
        },
        "required": [
          "field_name"
        ],
        "additionalProperties": False
      }
    }
  },
  "anyOf": [
    {
      "required": [
        "multiply_score_by"
      ]
    },
    {
      "required": [
        "add_to_score"
      ]
    }
  ],
  "additionalProperties": False
}
