import textwrap
import semver

from marqo.core.models import UnstructuredMarqoIndex
from marqo.core.models.marqo_index_request import UnstructuredMarqoIndexRequest
from marqo.core.unstructured_vespa_index import common as unstructured_common
from marqo.core.vespa_schema import VespaSchema


class UnstructuredVespaSchema(VespaSchema):
    _FIELD_ID = unstructured_common.VESPA_FIELD_ID

    _STRINGS = unstructured_common.STRINGS
    _SHORT_STRINGS_FIELDS = unstructured_common.SHORT_STRINGS_FIELDS
    _LONGS_STRINGS_FIELDS = unstructured_common.LONGS_STRINGS_FIELDS
    _STRING_ARRAY = unstructured_common.STRING_ARRAY

    _INT_FIELDS = unstructured_common.INT_FIELDS
    _FLOAT_FIELDS = unstructured_common.FLOAT_FIELDS
    _BOOL_FIELDS = unstructured_common.BOOL_FIELDS

    _SCORE_MODIFIERS = unstructured_common.SCORE_MODIFIERS

    _CHUNKS = unstructured_common.VESPA_DOC_CHUNKS
    _EMBEDDINGS = unstructured_common.VESPA_DOC_EMBEDDINGS

    _RANK_PROFILE_EMBEDDING_SIMILARITY = "embedding_similarity"
    _RANK_PROFILE_EMBEDDING_SIMILARITY_MODIFIERS = unstructured_common.RANK_PROFILE_EMBEDDING_SIMILARITY_MODIFIERS

    _QUERY_INPUT_EMBEDDING = "embedding_query"

    _SUMMARY_ALL_NON_VECTOR = 'all-non-vector-summary'
    _SUMMARY_ALL_VECTOR = 'all-vector-summary'

    def __init__(self, index_request: UnstructuredMarqoIndexRequest):
        self._index_request = index_request

    def generate_schema(self) -> (str, UnstructuredMarqoIndex):
        schema_name = self._get_vespa_schema_name(self._index_request.name)
        marqo_index = self._generate_unstructured_marqo_index(schema_name)
        unstructured_schema = self._generate_unstructured_schema(marqo_index)

        return unstructured_schema, marqo_index

    def _generate_unstructured_marqo_index(self, schema_name: str) -> UnstructuredMarqoIndex:
        """This function converts the attribute self._index_request: UnstructuredMarqoIndexRequest
        into an instance of UnstructuredMarqoIndex.
        """
        return UnstructuredMarqoIndex(
            name=self._index_request.name,
            schema_name=schema_name,
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
            filter_string_max_length=self._index_request.filter_string_max_length,
        )

    def _generate_unstructured_schema(self, marqo_index: UnstructuredMarqoIndex) -> str:
        """This function generates the Vespa schema for an unstructured Marqo index."""
        dimension = str(marqo_index.model.get_dimension())

        schema = textwrap.dedent(
            f"""
            schema {marqo_index.schema_name} {{
                document {{
                    field {self._FIELD_ID} type string {{
                        indexing: attribute | summary
                        attribute: fast-search
                        rank: filter
                    }}

                    field {self._STRINGS} type array<string>{{
                        indexing: index
                        index: enable-bm25
                    }}

                    field {self._LONGS_STRINGS_FIELDS} type map<string, string> {{
                        indexing: summary
                    }}

                    field {self._SHORT_STRINGS_FIELDS} type map<string, string> {{
                        indexing: summary
                        struct-field key {{ indexing : attribute
                                           attribute: fast-search
                                           rank: filter }}
                        struct-field value {{ indexing : attribute
                                              attribute: fast-search
                                              rank: filter }}
                    }}

                    field {self._STRING_ARRAY} type array<string> {{
                        indexing: attribute | summary
                        attribute: fast-search
                        rank: filter
                    }}
                    
                    field {unstructured_common.VESPA_DOC_MULTIMODAL_PARAMS} type map<string, string> {{
                        indexing: summary
                    }}

                    field {self._INT_FIELDS} type map<string, long> {{
                        indexing: summary
                        struct-field key {{ indexing : attribute
                                           attribute: fast-search
                                           rank: filter }}
                        struct-field value {{ indexing : attribute
                                           attribute: fast-search
                                           rank: filter }}
                    }}
                    
                    field {self._BOOL_FIELDS} type map<string, byte> {{
                        indexing: summary
                        struct-field key {{ indexing : attribute
                                            attribute: fast-search
                                            rank: filter }}
                        struct-field value {{ indexing : attribute
                                              attribute: fast-search
                                              rank: filter }}
                        }}
                                                    
                    field {self._FLOAT_FIELDS} type map<string, double> {{
                        indexing: summary
                        struct-field key {{ indexing : attribute
                                           attribute: fast-search
                                           rank: filter }}

                        struct-field value {{ indexing : attribute
                                           attribute: fast-search
                                           rank: filter }}
                    }}

                    field {self._SCORE_MODIFIERS} type tensor<double>(p{{}}) {{
                        indexing: attribute | summary
                    }}

                    field {self._CHUNKS} type array<string> {{
                        indexing: summary
                    }}
                    
                    field {unstructured_common.FIELD_VECTOR_COUNT} type int {{
                        indexing: attribute | summary
                    }}
                    
                    field {self._EMBEDDINGS} type tensor<float>(p{{}}, x[{dimension}]) {{
                        indexing: attribute | index | summary
                        attribute {{
                            distance-metric: {self._get_distance_metric(marqo_index.distance_metric)}
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
                    fields: {self._STRINGS}
                }}
            """
        )

        # Add rank profiles
        rank_profile_list = self._generate_rank_profiles(marqo_index)
        schema += "\n".join(rank_profile_list)

        # Add summaries
        schema += textwrap.dedent(
            f"""
                document-summary {self._SUMMARY_ALL_NON_VECTOR} {{
                    summary {self._FIELD_ID} type string {{}}
                    summary {self._STRINGS} type array<string> {{}}
                    summary {self._LONGS_STRINGS_FIELDS} type map<string, string> {{}}
                    summary {self._SHORT_STRINGS_FIELDS} type map<string, string> {{}}
                    summary {self._STRING_ARRAY} type array<string> {{}}
                    summary {self._BOOL_FIELDS} type map<string, byte> {{}}
                    summary {self._INT_FIELDS} type map<string, long> {{}}
                    summary {self._FLOAT_FIELDS} type map<string, double> {{}}
                    summary {self._CHUNKS} type array<string> {{}}
                }}

                document-summary {self._SUMMARY_ALL_VECTOR} {{
                    summary {self._FIELD_ID} type string {{}}
                    summary {self._STRINGS} type array<string> {{}}
                    summary {self._LONGS_STRINGS_FIELDS} type map<string, string> {{}}
                    summary {self._SHORT_STRINGS_FIELDS} type map<string, string> {{}}
                    summary {self._STRING_ARRAY} type array<string> {{}}
                    summary {self._BOOL_FIELDS} type map<string, byte> {{}}
                    summary {self._INT_FIELDS} type map<string, long> {{}}
                    summary {self._FLOAT_FIELDS} type map<string, double> {{}}
                    summary {self._CHUNKS} type array<string> {{}}
                    summary {self._EMBEDDINGS} type tensor<float>(p{{}}, x[{dimension}]) {{}}
                }}
            }}
            """
        )

        return schema

    def _generate_rank_profiles(self, marqo_index: UnstructuredMarqoIndex):
        rank_profiles: List[str] = list()
        model_dim = marqo_index.model.get_dimension()

        # generate base rank profile
        rank_profiles.extend(self._generate_base_rank_profile(marqo_index))

        # generate bm25 profile (from base)
        rank_profiles.append(f'rank-profile {unstructured_common.RANK_PROFILE_BM25} '
                             f'inherits {unstructured_common.RANK_PROFILE_BASE} {{')
        rank_profiles.append('first-phase {')
        rank_profiles.append(
            f'expression: modify(lexical_score(), '
            f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_LEXICAL}), '
            f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_LEXICAL})'
            f')'
        )
        rank_profiles.append('}}')

        # generate embedding similarity profile (from base)
        rank_profiles.append(f'rank-profile {unstructured_common.RANK_PROFILE_EMBEDDING_SIMILARITY} '
                             f'inherits {unstructured_common.RANK_PROFILE_BASE} {{')
        rank_profiles.append('first-phase {')
        rank_profiles.append(f'expression: modify(embedding_score(), '
                             f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_TENSOR}), '
                             f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_TENSOR})'
                             f')')
        rank_profiles.append('}')
        rank_profiles.append(f'match-features: closest({self._EMBEDDINGS})')
        rank_profiles.append('}')

        # hybrid custom searcher profile
        # Temp parameters to pass into respective queries (lexical and tensor)
        rank_profiles.append(f'rank-profile {unstructured_common.RANK_PROFILE_HYBRID_CUSTOM_SEARCHER} inherits default {{')
        rank_profiles.append('inputs {')
        rank_profiles.append(f'query({unstructured_common.QUERY_INPUT_EMBEDDING}) tensor<float>(x[{model_dim}])')
        rank_profiles.append(f'query({unstructured_common.QUERY_INPUT_HYBRID_FIELDS_TO_RANK_LEXICAL}) tensor<int8>(p{{}})')
        rank_profiles.append(f'query({unstructured_common.QUERY_INPUT_HYBRID_FIELDS_TO_RANK_TENSOR}) tensor<int8>(p{{}})')
        rank_profiles.append(
            f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_LEXICAL}) tensor<double>(p{{}})')
        rank_profiles.append(
            f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_LEXICAL}) tensor<double>(p{{}})')
        rank_profiles.append(
            f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_TENSOR}) tensor<double>(p{{}})')
        rank_profiles.append(
            f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_TENSOR}) tensor<double>(p{{}})')
        rank_profiles.append('}')
        rank_profiles.append('}')

        # hybrid bm25 and embedding_similarity rank profiles
        # HYBRID SEARCH LEXICAL THEN TENSOR
        rank_profiles.append(
            f'rank-profile {unstructured_common.RANK_PROFILE_HYBRID_BM25_THEN_EMBEDDING_SIMILARITY} inherits '
            f'{unstructured_common.RANK_PROFILE_BASE} {{'
        )

        rank_profiles.append('first-phase {')  # First phase ranks on LEXICAL
        rank_profiles.append(
            f'expression: modify(lexical_score(), '
            f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_LEXICAL}), '
            f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_LEXICAL}))'
        )
        rank_profiles.append('}')
        rank_profiles.append('second-phase {')  # Second phase ranks on TENSOR
        rank_profiles.append(
            f'expression: modify(embedding_score(), '
            f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_TENSOR}), '
            f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_TENSOR}))'
        )
        rank_profiles.append('}')
        rank_profiles.append(f'match-features: closest({self._EMBEDDINGS})')
        rank_profiles.append('}')

        # HYBRID SEARCH TENSOR THEN LEXICAL
        # Only 1 phase ranking (lexical only)
        rank_profiles.append(
            f'rank-profile {unstructured_common.RANK_PROFILE_HYBRID_EMBEDDING_SIMILARITY_THEN_BM25} inherits '
            f'{unstructured_common.RANK_PROFILE_BASE} {{'
        )
        rank_profiles.append('first-phase {')  # First phase ranks on LEXICAL
        rank_profiles.append(
            f'expression: modify(lexical_score(), '
            f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_LEXICAL}), '
            f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_LEXICAL}))')
        rank_profiles.append('}}')

        return rank_profiles

    def _generate_base_rank_profile(self, marqo_index: UnstructuredMarqoIndex):
        base_rank_profile: List[str] = list()
        model_dim = marqo_index.model.get_dimension()

        base_rank_profile.append(f'rank-profile {unstructured_common.RANK_PROFILE_BASE} inherits default {{')

        # inputs
        base_rank_profile.append('inputs {')
        base_rank_profile.append(f'query({unstructured_common.QUERY_INPUT_EMBEDDING}) tensor<float>(x[{model_dim}])')
        base_rank_profile.append(f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_LEXICAL}) tensor<double>(p{{}})')
        base_rank_profile.append(f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_LEXICAL}) tensor<double>(p{{}})')
        base_rank_profile.append(f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_MULT_WEIGHTS_TENSOR}) tensor<double>(p{{}})')
        base_rank_profile.append(f'query({unstructured_common.QUERY_INPUT_SCORE_MODIFIERS_ADD_WEIGHTS_TENSOR}) tensor<double>(p{{}})')
        base_rank_profile.append('}')

        # modifiers function
        score_modifier_expression = (
            f'if (count(mult_weights * attribute({self._SCORE_MODIFIERS})) == 0, '
            f'  1, reduce(mult_weights * attribute({self._SCORE_MODIFIERS}), prod)) '
            f'* score '
            f'+ reduce(add_weights * attribute({self._SCORE_MODIFIERS}), sum)'
        )
        base_rank_profile.append('function modify(score, mult_weights, add_weights) {')
        base_rank_profile.append(f'    expression: {score_modifier_expression}')
        base_rank_profile.append('}')

        # lexical score function
        base_rank_profile.append('function lexical_score() {')
        base_rank_profile.append(f'expression: bm25({self._STRINGS})')
        base_rank_profile.append('}')

        # embedding score function
        base_rank_profile.append('function embedding_score() {')
        base_rank_profile.append(f'expression: closeness(field, {self._EMBEDDINGS})')
        base_rank_profile.append('}')

        base_rank_profile.append('}')

        return base_rank_profile