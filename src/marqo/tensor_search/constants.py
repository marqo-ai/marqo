from marqo.tensor_search.enums import MappingsObjectType

MARQO_OBJECT_TYPES = {MappingsObjectType.multimodal_combination, MappingsObjectType.custom_vector}

ILLEGAL_CUSTOMER_FIELD_NAME_CHARS = {'.', '/', '\n'}

ALLOWED_UNSTRUCTURED_FIELD_TYPES = [str, int, float, bool, list]

NON_TENSORISABLE_FIELD_TYPES = [int, float, bool, list]

ALLOWED_MULTIMODAL_FIELD_TYPES = [str]

# Must be written in JSON schema type form: https://json-schema.org/understanding-json-schema/reference/type
ALLOWED_CUSTOM_VECTOR_CONTENT_TYPES = ["string"]
