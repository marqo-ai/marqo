context_schema = {
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "object",
  "properties": {
    "tensor": {
      "type": "array",
      "minItems":1,
      "maxItems" : 64,
      "items":
        {
          "type": "object",
          "properties": {
            "vector": {
              "type": "array",
              "items": {"type": "number"}
            },
            "weight": {
              "type": "number"
            }
          },
          "required": [
            "vector",
            "weight"
          ]
        },
    }
  },
  "required": [
    "tensor"
  ]
}