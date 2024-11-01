import os

from jinja2 import Environment, FileSystemLoader

from marqo.core.models.marqo_index import SemiStructuredMarqoIndex, MarqoIndex
from marqo.core.models.marqo_index_request import UnstructuredMarqoIndexRequest
from marqo.core.vespa_index.vespa_schema import VespaSchema


class SemiStructuredVespaSchema(VespaSchema):
    FIELD_INDEX_PREFIX = 'marqo__lexical_'
    FIELD_CHUNKS_PREFIX = 'marqo__chunks_'
    FIELD_EMBEDDING_PREFIX = 'marqo__embeddings_'

    def __init__(self, index_request: UnstructuredMarqoIndexRequest):
        self._index_request = index_request

    def generate_schema(self) -> (str, MarqoIndex):
        schema_name = self._get_vespa_schema_name(self._index_request.name)
        marqo_index = self._generate_marqo_index(schema_name)
        schema = self.generate_vespa_schema(marqo_index)
        return schema, marqo_index

    @classmethod
    def generate_vespa_schema(cls, marqo_index: SemiStructuredMarqoIndex) -> str:
        template_path = str(os.path.dirname(os.path.abspath(__file__)))
        environment = Environment(loader=FileSystemLoader(template_path))
        vespa_schema_template = environment.get_template("semi_structured_vespa_schema_template.sd.jinja2")
        return vespa_schema_template.render(index=marqo_index, dimension=str(marqo_index.model.get_dimension()))

    def _generate_marqo_index(self, schema_name: str) -> SemiStructuredMarqoIndex:
        marqo_index = SemiStructuredMarqoIndex(
            name=self._index_request.name,
            schema_name=schema_name,
            model=self._index_request.model,
            normalize_embeddings=self._index_request.normalize_embeddings,
            text_preprocessing=self._index_request.text_preprocessing,
            image_preprocessing=self._index_request.image_preprocessing,
            audio_preprocessing=self._index_request.audio_preprocessing,
            video_preprocessing=self._index_request.video_preprocessing,
            distance_metric=self._index_request.distance_metric,
            vector_numeric_type=self._index_request.vector_numeric_type,
            hnsw_config=self._index_request.hnsw_config,
            marqo_version=self._index_request.marqo_version,
            created_at=self._index_request.created_at,
            updated_at=self._index_request.updated_at,
            lexical_fields=[],
            tensor_fields=[],
            filter_string_max_length=self._index_request.filter_string_max_length,
            treat_urls_and_pointers_as_images=self._index_request.treat_urls_and_pointers_as_images,
            treat_urls_and_pointers_as_media=self._index_request.treat_urls_and_pointers_as_media,
        )

        return marqo_index
