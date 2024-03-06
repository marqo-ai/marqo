from marqo.tensor_search.enums import MappingsObjectType
from marqo.tensor_search.constants import ALLOWED_CUSTOM_VECTOR_CONTENT_TYPES
from marqo.base_model import StrictBaseModel
from typing import List, Union, Optional
from pydantic import root_validator, validator


class CustomVector(StrictBaseModel):
    """
    Required structure for a custom vector object
    """
    dict_data: dict
    dimension: int
    is_non_tensor_field: bool

    @root_validator
    def validate_custom_vector(cls, values):
        dict_data = values.get('dict_data', [])
        dimension = values.get('dimension')
        is_non_tensor_field = values.get('is_non_tensor_field')

        # Must be tensor field
        if is_non_tensor_field:
            raise ValueError(
                f"Cannot create custom_vector field (given field content: '{dict_data}') as a non-tensor field. "
                f"Add this field to 'tensor_fields' to fix this problem."
            )

        # Extra fields are not allowed
        extra_fields = dict_data.keys() - {"vector", "content"}
        if extra_fields:
            raise ValueError(f"Custom vector field can only contain fields 'vector' and 'content'. Received unexpected extra fields: {extra_fields}.")

        # Vector must be present in dict
        if "vector" in dict_data:
            # Validate vector is a list of numbers
            vector = dict_data["vector"]
            if not isinstance(vector, List):
                raise ValueError(f"Custom vector field 'vector' must be a list, but given vector is of type {type(vector)}")
            for element in vector:
                if not isinstance(element, (int, float)):
                    raise ValueError(
                        f"Custom vector field 'vector' must be a list of numbers, but given vector contains an element of type {type(element)}"
                    )
        else:
            raise ValueError(f"Cannot create custom_vector field missing 'vector', which is a required key. "
                             f"Given field content: '{dict_data}'")

        # Vector length must be correct
        if dimension and len(vector) != dimension:
            raise ValueError(f"Custom vector must have dimension of {dimension}, but given vector is of length {len(vector)}")

        # Fill in content with default if not present
        if "content" in dict_data:
            # Verify content is an allowed type
            content_is_allowed_type = False
            for allowed_type in ALLOWED_CUSTOM_VECTOR_CONTENT_TYPES:
                if isinstance(dict_data["content"], allowed_type):
                    content_is_allowed_type = True
                    break
            if not content_is_allowed_type:
                raise ValueError(
                    f"Custom vector field 'content' must be one of the following types: {ALLOWED_CUSTOM_VECTOR_CONTENT_TYPES}, "
                    f"but given content is of type {type(dict_data['content'])}"
                )
        else:
            # Transform dict: Fill in content with empty string by default
            values["dict_data"]["content"] = ""

        return values

    def to_dict(self):
        return self.dict_data