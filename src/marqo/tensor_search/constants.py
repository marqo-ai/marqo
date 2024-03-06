from marqo.tensor_search.enums import MappingsObjectType

MARQO_OBJECT_TYPES = {MappingsObjectType.multimodal_combination, MappingsObjectType.custom_vector}

ILLEGAL_CUSTOMER_FIELD_NAME_CHARS = {'.', '/', '\n'}

ALLOWED_UNSTRUCTURED_FIELD_TYPES = [str, int, float, bool, list, dict]

NON_TENSORISABLE_FIELD_TYPES = [int, float, bool, list]

ALLOWED_MULTIMODAL_FIELD_TYPES = [str]

ALLOWED_CUSTOM_VECTOR_CONTENT_TYPES = [str]
