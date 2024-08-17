from pydantic import Field

from marqo.base_model import ImmutableStrictBaseModel


class TensorField(ImmutableStrictBaseModel):
    """
    A tensor field that has a corresponding field.

    chunk_field_name and embeddings_field_name must be unique across all tensor fields.
    """
    _FIELD_CHUNKS_PREFIX = 'marqo__chunks_'
    _FIELD_EMBEDDING_PREFIX = 'marqo__embeddings_'

    name: str
    chunk_field_name: str = Field(None)
    embeddings_field_name: str = Field(None)

    def __init__(self, **data):
        if 'chunk_field_name' not in data:
            data['chunk_field_name'] = self._FIELD_CHUNKS_PREFIX + data['name']
        if 'embeddings_field_name' not in data:
            data['embeddings_field_name'] = self._FIELD_EMBEDDING_PREFIX + data['name']
        super().__init__(**data)

