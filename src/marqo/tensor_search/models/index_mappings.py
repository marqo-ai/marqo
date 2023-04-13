import copy
from typing import List



class IndexMappings:

    def __init__(self, index_mappings: dict):
        self.index_mappings = copy.deepcopy(index_mappings)

    def get_score_modifier_fields(self) -> List:
        pass

    def generate_appending_vectors(self, doc) -> List:
        pass
