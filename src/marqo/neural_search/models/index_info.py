import pprint
from typing import NamedTuple, Any
from marqo.neural_search import enums


class IndexInfo(NamedTuple):
    """
    model_name: name of the ML model used to encode the data
    properties: keys are different index field names, values
        provide info about the properties
    """
    model_name: str
    properties: dict
    neural_settings: dict

    def get_neural_settings(self) -> dict:
        return self.neural_settings.copy()

    def get_vector_properties(self) -> dict:
        """returns a dict containing only names and properties of vector fields
        Perhaps a better approach is to check if the field's props is actually a vector type,
        plus checks over fieldnames
        """
        return {
            vector_name: vector_props
            for vector_name, vector_props in self.properties[enums.NeuralField.chunks]["properties"].items()
            if vector_name.startswith(enums.NeuralField.vector_prefix)
        }

    def get_text_properties(self):
        """returns a dict containing only names and properties of text fields
        Perhaps a better approach is to check if the text's props is actually a text type,
        plus checks over fieldnames"""
        return {
            text_field: text_props
            for text_field, text_props in self.properties.items()
            if not text_field.startswith(enums.NeuralField.vector_prefix)
                and not text_field in enums.NeuralField.__dict__.values()
        }