import pprint
from typing import NamedTuple, Any, Dict
from marqo.tensor_search import enums
from marqo.tensor_search.enums import IndexSettingsField as NsFields
from marqo.tensor_search import configs

class IndexInfo(NamedTuple):
    """
    model_name: name of the ML model used to encode the data
    properties: keys are different index field names, values
        provide info about the properties
    """
    model_name: str
    properties: dict
    index_settings: dict

    def get_index_settings(self) -> dict:
        return self.index_settings.copy()

    def get_vector_properties(self) -> dict:
        """returns a dict containing only names and properties of vector fields
        Perhaps a better approach is to check if the field's props is actually a vector type,
        plus checks over fieldnames
        """
        return {
            vector_name: vector_props
            for vector_name, vector_props in self.properties[enums.TensorField.chunks]["properties"].items()
            if vector_name.startswith(enums.TensorField.vector_prefix)
        }

    def get_text_properties(self) -> dict:
        """returns a dict containing only names and properties of non
        vector fields.

        This returns more than just pure text fields. For example: ints
        bool fields.

        """
        text_props_dict = {}
        for text_field, text_props in self.properties.items():
            if not text_field.startswith(enums.TensorField.vector_prefix) and not text_field in enums.TensorField.__dict__.values():
                if not "properties" in text_props:
                    text_props_dict[text_field] = text_props
                elif "properties" in text_props:
                    for sub_field, sub_field_props in text_props["properties"].items():
                            text_props_dict[f"{text_field}.{sub_field}"] = sub_field_props
        return text_props_dict

        # get_text_properties will flatten the object properties
        # Example: left-text_properties   right-true_text_properties
        # {'Description': {'type': 'text'},                                                         {'Description': {'type': 'text'},
        #  'Genre': {'type': 'text'},                                                                'Genre': {'type': 'text'},
        #  'Title': {'type': 'text'},                                                                'Title': {'type': 'text'},
        #  'my_combination_field': {'properties': {'lexical_field': {'type': 'text'}, ----->         'my_combination_field.lexical_field': {'type': 'text'},
        #                                          'my_image': {'type': 'text'},                     'my_combination_field.my_image': {'type': 'text'},
        #                                          'some_text': {'type': 'text'}}}}                   'my_combination_field.some_text': {'type': 'text'}}

    def get_true_text_properties(self) -> dict:
        """returns a dict containing only names and properties of fields that
        are true text fields
        """
        simple_props = self.get_text_properties()
        true_text_props = dict()
        for text_field, text_props in simple_props.items():
            try:
                if text_props["type"] == enums.OpenSearchDataType.text:
                    true_text_props[text_field] = text_props
            except KeyError:
                continue
        return true_text_props
    
    def get_ann_parameters(self) -> Dict[str, Any]:
        """Gets the ANN parameters to use as the default for the index.

        Preferentially use index settings (when present) over generic defaults.

        Returns:
            Dict of ann parameters. Structure can be seen at `configs.get_default_ann_parameters`.
        
        """
        ann_default = configs.get_default_ann_parameters()
        index_ann_defaults = self.index_settings[NsFields.index_defaults].get(NsFields.ann_parameters, {})

        # index defaults override generic defaults
        ann_params = {
            **ann_default,
            **index_ann_defaults
        }
        ann_params[NsFields.ann_method_parameters] = {
            **ann_default[NsFields.ann_method_parameters],
            **index_ann_defaults.get(NsFields.ann_method_parameters, {})
        }

        return ann_params