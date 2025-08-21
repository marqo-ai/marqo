import os
from jinja2 import Environment, FileSystemLoader
from marqo.core.vespa_index.vespa_schema import VespaSchema


class TypeaheadVespaSchema:
    """Schema generator for typeahead functionality."""

    def __init__(self, index_name: str):
        self.index_name = index_name

    def generate_schema(self) -> (str, str):
        """
        Generate a Vespa schema for typeahead functionality.
        
        Returns:
            A tuple containing the Vespa schema name and the schema definition as a string.
        """
        schema_name = self._get_typeahead_schema_name(self.index_name)

        template_path = str(os.path.dirname(os.path.abspath(__file__)))
        environment = Environment(loader=FileSystemLoader(template_path))
        vespa_schema_template = environment.get_template("typeahead_vespa_schema_template.sd.jinja2")

        return self._get_typeahead_schema_name(self.index_name), vespa_schema_template.render(schema_name=schema_name)

    def _get_typeahead_schema_name(self, index_name: str) -> str:
        """
        Get the name of the typeahead Vespa schema.

        Args:
            index_name: Name of the main index

        Returns:
            Name for the typeahead schema
        """
        # Note our encoding means it's impossible to have another index's schem name clash with this. This is because
        # no index name leads to _typeahead as _ itself is encoded to _00
        encoded_name = self._get_vespa_schema_name(index_name)
        return f"{encoded_name}_typeahead"