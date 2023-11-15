import textwrap

from marqo.core.models import MarqoIndex, UnstructuredMarqoIndex
from marqo.core.models.marqo_index_request import UnstructuredMarqoIndexRequest
from marqo.core.vespa_schema import VespaSchema


class UnstructuredVespaSchema(VespaSchema):
    _FIELD_ID = "marqo__id"

    _STRINGS = "marqo__strings"
    _SHORT_STRINGS_FIELDS = "marqo__short_strings_fields"
    _LONGS_STRINGS_FIELDS = "marqo__long_strings_fields"
    _STRING_ARRAY = "marqo__string_array"

    _INT_FIELDS = "marqo__int_fields"
    _FLOAT_FIELDS = "marqo__float_fields"

    _SCORE_MODIFIERS = "marqo__score_modifiers"

    _CHUNKS = "marqo__chunks"
    _EMBEDDINGS = "marqo__embeddings"

    _RANK_PROFILE_EMBEDDING_SIMILARITY = "embedding_similarity"
    _QUERY_INPUT_EMBEDDING = "embedding_query"

    _SUMMARY_ALL_NON_VECTOR = 'all-non-vector-summary'
    _SUMMARY_ALL_VECTOR = 'all-vector-summary'

    def __init__(self, index_request: UnstructuredMarqoIndexRequest):
        self._index_request = index_request

    def generate_schema(self) -> (str, UnstructuredMarqoIndex):
        marqo_index = self._generate_unstructured_marqo_index()
        unstructured_schema = self._generate_unstructured_schema(marqo_index)

        return unstructured_schema, marqo_index

    def _generate_unstructured_marqo_index(self) -> UnstructuredMarqoIndex:
        """This function converts the attribute self._index_request: UnstructuredMarqoIndexRequest
        into an instance of UnstructuredMarqoIndex.
        """
        return UnstructuredMarqoIndex(
            name=self._index_request.name,
            model=self._index_request.model,
            normalize_embeddings=self._index_request.normalize_embeddings,
            text_preprocessing=self._index_request.text_preprocessing,
            image_preprocessing=self._index_request.image_preprocessing,
            distance_metric=self._index_request.distance_metric,
            vector_numeric_type=self._index_request.vector_numeric_type,
            hnsw_config=self._index_request.hnsw_config,
            marqo_version=self._index_request.marqo_version,
            created_at=self._index_request.created_at,
            updated_at=self._index_request.updated_at,
            treat_urls_and_pointers_as_images=self._index_request.treat_urls_and_pointers_as_images,
        )

    @classmethod
    def _generate_unstructured_schema(cls, marqo_index: UnstructuredMarqoIndex) -> str:
        """This function generates the Vespa schema for an unstructured Marqo index."""
        dimension = str(marqo_index.model.get_dimension())

        return textwrap.dedent(
            f"""
            schema {marqo_index.name} {{
                document {marqo_index.name} {{
                    field {cls._FIELD_ID} type string {{
                        indexing: attribute | summary
                    }}

                    field {cls._STRINGS} type array<string>{{
                        indexing: index
                        index: enable-bm25
                    }}

                    field {cls._LONGS_STRINGS_FIELDS} type map<string, string> {{
                        indexing: summary
                    }}

                    field {cls._SHORT_STRINGS_FIELDS} type map<string, string> {{
                        indexing: summary
                        struct-field key {{ indexing : attribute
                                           attribute: fast-search
                                           rank: filter }}
                        struct-field value {{ indexing : attribute
                           attribute: fast-search
                          rank: filter }}
                        dictionary: hash
                    }}

                    field {cls._STRING_ARRAY} type array<string> {{
                        indexing: attribute | summary
                        attribute: fast-search
                        rank: filter
                    }}

                    field {cls._INT_FIELDS} type map<string, int> {{
                        indexing: summary
                        struct-field key {{ indexing : attribute
                                           attribute: fast-search
                                           rank: filter }}
                        struct-field value {{ indexing : attribute
                                           attribute: fast-search
                                           rank: filter }}
                        dictionary: hash
                    }}

                    field {cls._FLOAT_FIELDS} type map<string, float> {{
                        indexing: summary
                        struct-field key {{ indexing : attribute
                                           attribute: fast-search
                                           rank: filter }}

                        struct-field value {{ indexing : attribute
                                           attribute: fast-search
                                           rank: filter }}
                        dictionary: hash
                    }}


                    field {cls._SCORE_MODIFIERS} type tensor<float>(p{{}}) {{
                        indexing: attribute
                    }}

                    field {cls._CHUNKS} type array<string> {{
                        indexing: attribute | summary
                    }}

                    field {cls._EMBEDDINGS} type tensor<float>(p{{}}, x[{dimension}]) {{
                        indexing: attribute | index | summary
                        attribute {{
                            distance-metric: prenormalized-angular
                        }}
                        index {{
                            hnsw {{
                                max-links-per-node: {marqo_index.hnsw_config.m}
                                neighbors-to-explore-at-insert: {marqo_index.hnsw_config.ef_construction}
                            }}
                        }}
                    }}
                }}

                fieldset default {{
                    fields: {cls._STRINGS}
                }}

                rank-profile {cls._RANK_PROFILE_EMBEDDING_SIMILARITY} inherits default {{
                    inputs {{
                        query({cls._QUERY_INPUT_EMBEDDING}) tensor<float>(x[{dimension}])
                    }}
                    first-phase {{
                        expression: closeness(field, {cls._EMBEDDINGS})
                    }}
                    match-features: closest({cls._EMBEDDINGS})
                }}

                rank-profile bm25 inherits default {{
                    first-phase {{
                    expression: bm25({cls._STRINGS})
                    }}
                }}

                document-summary {cls._SUMMARY_ALL_NON_VECTOR} {{
                    summary {cls._FIELD_ID} type string {{}}
                    summary {cls._STRINGS} type array<string> {{}}
                    summary {cls._LONGS_STRINGS_FIELDS} type map<string, string> {{}}
                    summary {cls._SHORT_STRINGS_FIELDS} type map<string, string> {{}}
                    summary {cls._STRING_ARRAY} type array<string> {{}}
                    summary {cls._INT_FIELDS} type map<string, int> {{}}
                    summary {cls._FLOAT_FIELDS} type map<string, float> {{}}
                    summary {cls._CHUNKS} type array<string> {{}}
                }}

                document-summary {cls._SUMMARY_ALL_VECTOR} {{
                    summary {cls._FIELD_ID} type string {{}}
                    summary {cls._STRINGS} type array<string> {{}}
                    summary {cls._LONGS_STRINGS_FIELDS} type map<string, string> {{}}
                    summary {cls._SHORT_STRINGS_FIELDS} type map<string, string> {{}}
                    summary {cls._STRING_ARRAY} type array<string> {{}}
                    summary {cls._INT_FIELDS} type map<string, int> {{}}
                    summary {cls._FLOAT_FIELDS} type map<string, float> {{}}
                    summary {cls._EMBEDDINGS} type tensor<float>(p{{}}, x[{dimension}]) {{}}
                }}
            }}
            """
        )