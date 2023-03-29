from marqo.tensor_search.constants import ALLOWED_CUSTOM_SCORE_FIELDS_OPERATIONS

custom_score_fiedls_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["field_name", "weight", "combine_style"],
  "properties": {
    "field_name": {
      "type": "string"
    },
    "weight": {
      "type": "number"
    },
    "combine_style": {
      "type": "string",
      "enum": ALLOWED_CUSTOM_SCORE_FIELDS_OPERATIONS,
    }
  }
}

