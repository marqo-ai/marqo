context_schema = {
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "object",
  "properties": {
    "tensor": {
      "type": "array",
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