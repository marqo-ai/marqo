import os

from jinja2 import Environment, FileSystemLoader

from marqo.core.models import MarqoIndex


class TypeaheadVespaSchema:
    """Schema generator for typeahead functionality."""

    def __init__(self, marqo_index: MarqoIndex):
        self.marqo_index = marqo_index

    def generate_schema(self) -> (str, MarqoIndex):
        """
        Generate a Vespa schema for typeahead functionality.
        
        Returns:
            A tuple containing the Vespa schema name and the schema definition as a string.
        """
        typeahead_schema_name = self._get_typeahead_schema_name(self.marqo_index.schema_name)

        template_path = str(os.path.dirname(os.path.abspath(__file__)))
        environment = Environment(loader=FileSystemLoader(template_path))
        vespa_schema_template = environment.get_template("typeahead_vespa_schema_template.sd.jinja2")

        new_marqo_index = self.marqo_index.copy(deep=True, update={"typeahead_schema_name": typeahead_schema_name})

        return vespa_schema_template.render(schema_name=typeahead_schema_name), new_marqo_index

    def _get_typeahead_schema_name(self, schema_name: str) -> str:
        """
        Get the name of the typeahead Vespa schema.

        Args:
            index_name: Name of the main index

        Returns:
            Name for the typeahead schema
        """
        # Note our encoding means it's impossible to have another index's schem name clash with this. This is because
        # no index name leads to _typeahead as _ itself is encoded to _00
        return f"{schema_name}_typeahead"
