import copy
from typing import List



class IndexMappings:
    def __init__(self, index_mappings: dict):
        self.index_mappings = copy.deepcopy(index_mappings)
        self.score_modifiers_fields = self.get_score_modifier_fields()

    def get_score_modifier_fields(self) -> List:
        score_modifier_fields = []
        for field_name, field_attributes in self.index_mappings.items():
            if field_attributes["type"] == "score_modifier_field":
                score_modifier_fields.append(field_name)
        return score_modifier_fields

    def generate_appending_vectors(self, doc) -> List:
        appending_vector = [0.0,] * len(self.score_modifiers_fields)
        for field_name in list(doc):
            if field_name in self. score_modifiers_fields and isinstance(doc[field_name],  (int, float)):
                appending_vector[self.index_mappings[field_name]["appending_vector_position"]] = \
                    doc[field_name] * self.index_mappings[field_name]["scale_factor"]

        return appending_vector
