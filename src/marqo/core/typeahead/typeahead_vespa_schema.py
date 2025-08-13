import os
from jinja2 import Environment, FileSystemLoader
from marqo.core.vespa_index.vespa_schema import VespaSchema


class TypeaheadVespaSchema(VespaSchema):
    """Schema generator for typeahead functionality."""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
    
    def generate_schema(self) -> str:
        """
        Generate a Vespa schema for typeahead functionality.
        
        Returns:
            A string containing the Vespa schema
        """
        schema_name = self._get_typeahead_schema_name(self.index_name)
        
        template_path = str(os.path.dirname(os.path.abspath(__file__)))
        environment = Environment(loader=FileSystemLoader(template_path))
        vespa_schema_template = environment.get_template("typeahead_vespa_schema_template.sd.jinja2")
        
        return vespa_schema_template.render(schema_name=schema_name)
    
    def _get_typeahead_schema_name(self, index_name: str) -> str:
        """
        Get the name of the typeahead Vespa schema.
        
        Args:
            index_name: Name of the main index
            
        Returns:
            Name for the typeahead schema
        """
        # Encode index name and add typeahead suffix
        encoded_name = self._encode_index_name(index_name)
        return f"{encoded_name}__typeahead"
    
    def _encode_index_name(self, index_name: str) -> str:
        """Encode index name for Vespa schema naming."""
        encoded = index_name
        for char, replacement in self._INDEX_NAME_ENCODING_MAP.items():
            encoded = encoded.replace(char, replacement)
        return encoded