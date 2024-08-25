from typing import Dict, Any, Optional, List, Union

import marqo.core.search.search_filter as search_filter
from marqo.core import constants
from marqo.core.exceptions import (InvalidDataTypeError, InvalidFieldNameError, VespaDocumentParsingError,
                                   InvalidDataRangeError, MarqoDocumentParsingError)
from marqo.core.models import MarqoQuery
from marqo.core.models.hybrid_parameters import RankingMethod, RetrievalMethod
from marqo.core.models.marqo_index import FieldType, FieldFeature, Field, logger, SemiStructuredMarqoIndex
from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoLexicalQuery, MarqoHybridQuery
from marqo.core.semi_structured_vespa_index.semi_structured_document import SemiStructuredVespaDocument
from marqo.core.structured_vespa_index import common
from marqo.core.vespa_index import VespaIndex
from marqo.exceptions import InternalError, InvalidArgumentError
import semver

from marqo.core.unstructured_vespa_index import common as unstructured_common

class SemiStructuredVespaIndex(VespaIndex):
    """
    An implementation of VespaIndex for structured indexes.
    """

    _MARQO_TO_PYTHON_TYPE_MAP = {
        FieldType.Text: str,
        FieldType.Bool: bool,
        FieldType.Int: int,
        FieldType.Long: int,
        FieldType.Float: [float, int],
        FieldType.Double: [float, int],
        FieldType.ArrayText: (list, str),
        FieldType.ArrayInt: (list, int),
        FieldType.ArrayFloat: (list, (float, int)),
        FieldType.ArrayLong: (list, int),
        FieldType.ArrayDouble: (list, (float, int)),
        FieldType.ImagePointer: str,
        FieldType.MultimodalCombination: dict,
        FieldType.CustomVector: str,
        FieldType.MapInt: (dict, int),
        FieldType.MapFloat: (dict, float),
        FieldType.MapDouble: (dict, float),
        FieldType.MapLong: (dict, int)
    }

    _VESPA_DOC_ID = 'id'
    _VESPA_DOC_FIELDS = 'fields'
    _VESPA_DOC_RELEVANCE = 'relevance'
    _VESPA_DOC_MATCH_FEATURES = 'matchfeatures'
    _VESPA_DOC_FIELDS_TO_IGNORE = {'sddocname'}

    _DEFAULT_MAX_LIMIT = 1000
    _DEFAULT_MAX_OFFSET = 10000

    _MAX_FLOAT = 3.4028235e38
    _MIN_FLOAT = -3.4028235e38

    _MAX_INT = 2147483647
    # The actual minimum value is -2147483648, but we use -2147483647 as this is the minimum to support filtering
    _MIN_INT = -2147483647

    _MAX_LONG = 9223372036854775807
    _MIN_LONG = -9223372036854775808

    _FILTER_STRING_BOOL_VALUES = ["true", "false"]
    _RESERVED_FIELD_SUBSTRING = "::"

    _HYBRID_SEARCH_MINIMUM_VERSION = semver.VersionInfo.parse(constants.MARQO_STRUCTURED_HYBRID_SEARCH_MINIMUM_VERSION)

    def get_vespa_id_field(self) -> str:
        return common.FIELD_ID

    def to_vespa_partial_document(self, marqo_document: Dict[str, Any]) -> Dict[str, Any]:
        vespa_id: Optional[str] = None
        vespa_fields: Dict[str, Any] = dict()
        score_modifiers_2_8: Dict[str, float] = dict()
        score_modifiers_float: Dict[str, float] = {}
        score_modifiers_double_long: Dict[str, float] = {}

        if constants.MARQO_DOC_ID not in marqo_document:
            raise MarqoDocumentParsingError(f"'{constants.MARQO_DOC_ID}' is a required field but it does not exist")
        else:
            vespa_id = marqo_document[constants.MARQO_DOC_ID]
            self._verify_id_field(vespa_id)

        for marqo_field in marqo_document:
            if marqo_field == constants.MARQO_DOC_ID:
                continue
            if marqo_field == constants.MARQO_DOC_TENSORS:
                raise MarqoDocumentParsingError(f"You cannot modify '{marqo_field}' field. ")

            tensor_fields_names = [tensor_field.name for tensor_field in self._marqo_index.tensor_fields]
            if marqo_field in tensor_fields_names:
                raise MarqoDocumentParsingError(f"You cannot modify '{marqo_field}' field as this is a tensor field")

            dependent_fields_names = self._marqo_index.dependent_fields_names
            if marqo_field in dependent_fields_names:
                raise MarqoDocumentParsingError(f"You cannot modify '{marqo_field}' "
                                                f"field as this is a dependent field of a multimodal combination field")

            marqo_value = marqo_document[marqo_field]
            self._verify_marqo_field_name(marqo_field)
            self._verify_marqo_field_type(marqo_field, marqo_value)

            index_field = self._marqo_index.field_map[marqo_field]

            if (not isinstance(marqo_value, bool)) and isinstance(marqo_value, (int, float)):
                self._verify_numerical_field_value(marqo_value, index_field)

            if isinstance(marqo_value, list) and len(marqo_value) > 0 and type(marqo_value[0]) in (float, int):
                for v in marqo_value:
                    self._verify_numerical_field_value(v, index_field)

            if isinstance(marqo_value, dict) and index_field.type in (FieldType.MapFloat, FieldType.MapInt,
                                                                      FieldType.MapLong, FieldType.MapDouble):
                for k, v in marqo_value.items():
                    if type(v) in (float, int):
                        self._verify_numerical_field_value(v, index_field)

            if index_field.type == FieldType.Bool:
                # Booleans are stored as bytes in Vespa
                marqo_value = int(marqo_value)

            if index_field.lexical_field_name:
                vespa_fields[index_field.lexical_field_name] = {
                    "assign": marqo_value
                }
            if index_field.filter_field_name:
                vespa_fields[index_field.filter_field_name] = {
                    "assign": marqo_value
                }
            if not index_field.lexical_field_name and not index_field.filter_field_name:
                vespa_fields[index_field.name] = {
                    "assign": marqo_value
                }

            if FieldFeature.ScoreModifier in index_field.features:
                if self._marqo_index_version < self._VERSION_2_9_0:
                    target_dict = score_modifiers_2_8
                else:
                    if index_field.type in [FieldType.MapFloat, FieldType.Float]:
                        target_dict = score_modifiers_float
                    else:
                        target_dict = score_modifiers_double_long

                if isinstance(marqo_value, dict):
                    for key, value in marqo_value.items():
                        target_dict[f'{index_field.name}.{key}'] = value
                else:
                    target_dict[index_field.name] = marqo_value

        if len(score_modifiers_double_long) > 0:
            vespa_fields[common.FIELD_SCORE_MODIFIERS_DOUBLE_LONG] = {
                "modify": {
                    "operation": "replace",
                    "cells": score_modifiers_double_long
                }
            }

        if len(score_modifiers_float) > 0:
            vespa_fields[common.FIELD_SCORE_MODIFIERS_FLOAT] = {
                "modify": {
                    "operation": "replace",
                    "cells": score_modifiers_float
                }
            }

        if len(score_modifiers_2_8) > 0:
            vespa_fields[common.FIELD_SCORE_MODIFIERS_2_8] = {
                "modify": {
                    "operation": "replace",
                    "cells": score_modifiers_2_8
                }
            }

        return {"id": vespa_id, "fields": vespa_fields}


    def to_vespa_document(self, marqo_document: Dict[str, Any]) -> Dict[str, Any]:
        unstructured_document: SemiStructuredVespaDocument = \
            (SemiStructuredVespaDocument.from_marqo_document(marqo_document, marqo_index=self._marqo_index))
        return unstructured_document.to_vespa_document()

    def to_marqo_document(self, vespa_document: Dict[str, Any], return_highlights: bool = False) -> Dict[str, Any]:
        unstructured_document: SemiStructuredVespaDocument = SemiStructuredVespaDocument.from_vespa_document(vespa_document, marqo_index=self._marqo_index)
        return unstructured_document.to_marqo_document(marqo_index=self._marqo_index, return_highlights=return_highlights)

    def to_vespa_query(self, marqo_query: MarqoQuery) -> Dict[str, Any]:
        # TODO - There is some inefficiency here, as we are retrieving chunks even if highlights are false,
        # and also for lexical search. This applies to both with and without attributes_to_retrieve

        # Verify attributes to retrieve, if defined
        if marqo_query.attributes_to_retrieve is not None:
            chunk_field_names = []
            for att in marqo_query.attributes_to_retrieve:
                if att not in self._marqo_index.field_map:
                    raise InvalidFieldNameError(
                        f'Index {self._marqo_index.name} has no field {att}. '
                        f'Available fields are: {", ".join(self._marqo_index.field_map.keys())}'
                    )
                if att in self._marqo_index.tensor_field_map:
                    chunk_field_names.append(self._marqo_index.tensor_field_map[att].chunk_field_name)

            marqo_query.attributes_to_retrieve.append(common.FIELD_ID)
            marqo_query.attributes_to_retrieve.extend(chunk_field_names)

        # Verify score modifiers, if defined
        if marqo_query.score_modifiers is not None:
            for modifier in marqo_query.score_modifiers:
                if '.' in modifier.field:
                    root_modifier_field, subfield = modifier.field.split('.')
                else:
                    root_modifier_field = modifier.field
                if root_modifier_field not in self._marqo_index.score_modifier_fields_names:
                    raise InvalidFieldNameError(
                        f'Index {self._marqo_index.name} has no score modifier field {modifier.field}. '
                        f'Available score modifier fields are: '
                        f'{", ".join(self._marqo_index.score_modifier_fields_names)}'
                    )

        # Hybrid must be checked first since it is a subclass of Tensor and Lexical
        if isinstance(marqo_query, MarqoHybridQuery):
            return self._to_vespa_hybrid_query(marqo_query)
        elif isinstance(marqo_query, MarqoTensorQuery):
            return self._to_vespa_tensor_query(marqo_query)
        elif isinstance(marqo_query, MarqoLexicalQuery):
            return self._to_vespa_lexical_query(marqo_query)

        else:
            raise InternalError(f'Unknown query type {type(marqo_query)}')

    def get_vector_count_query(self):
        return {
            'yql': f'select {common.FIELD_VECTOR_COUNT} from {self._marqo_index.schema_name} '
                   f'where true limit 0 | all(group(1) each(output(sum({common.FIELD_VECTOR_COUNT}))))',
            'model_restrict': self._marqo_index.schema_name,
            'timeout': '5s'
        }

    def _to_vespa_tensor_query(self, marqo_query: MarqoTensorQuery) -> Dict[str, Any]:
        fields_to_search = self._get_tensor_fields_to_search(marqo_query)

        tensor_term = self._get_tensor_search_term(marqo_query) if fields_to_search else "False"
        filter_term = self._get_filter_term(marqo_query)
        if filter_term:
            filter_term = f' AND {filter_term}'
        else:
            filter_term = ''
        select_attributes = self._get_select_attributes(marqo_query)
        summary = common.SUMMARY_ALL_VECTOR if marqo_query.expose_facets else common.SUMMARY_ALL_NON_VECTOR
        score_modifiers = self._get_score_modifiers(marqo_query)

        if self._marqo_index_version < self._HYBRID_SEARCH_MINIMUM_VERSION:
            ranking = common.RANK_PROFILE_EMBEDDING_SIMILARITY_MODIFIERS_2_9 if score_modifiers \
                else common.RANK_PROFILE_EMBEDDING_SIMILARITY
        else:
            ranking = common.RANK_PROFILE_EMBEDDING_SIMILARITY

        query_inputs = {
            common.QUERY_INPUT_EMBEDDING: marqo_query.vector_query
        }
        query_inputs.update({
            f: 1 for f in fields_to_search
        })
        if score_modifiers:
            query_inputs.update(score_modifiers)

        query = {
            'yql': f'select {select_attributes} from {self._marqo_index.schema_name} where {tensor_term}{filter_term}',
            'model_restrict': self._marqo_index.schema_name,
            'hits': marqo_query.limit,
            'offset': marqo_query.offset,
            'query_features': query_inputs,
            'presentation.summary': summary,
            'ranking': ranking
        }
        query = {k: v for k, v in query.items() if v is not None}

        if not marqo_query.approximate:
            query['ranking.softtimeout.enable'] = False
            query['timeout'] = 300 * 1000  # 5 minutes

        return query

    def _to_vespa_lexical_query(self, marqo_query: MarqoLexicalQuery) -> Dict[str, Any]:
        fields_to_search = self._get_lexical_fields_to_search(marqo_query)

        lexical_term = self._get_lexical_search_term(marqo_query) if fields_to_search else "False"
        filter_term = self._get_filter_term(marqo_query)
        if filter_term:
            search_term = f'({lexical_term}) AND ({filter_term})'
        else:
            search_term = f'({lexical_term})'

        select_attributes = self._get_select_attributes(marqo_query)
        summary = common.SUMMARY_ALL_VECTOR if marqo_query.expose_facets else common.SUMMARY_ALL_NON_VECTOR
        score_modifiers = self._get_score_modifiers(marqo_query)

        if self._marqo_index_version < self._HYBRID_SEARCH_MINIMUM_VERSION:
            ranking = common.RANK_PROFILE_BM25_MODIFIERS_2_9 if score_modifiers \
                else common.RANK_PROFILE_BM25
        else:
            ranking = common.RANK_PROFILE_BM25

        query_inputs = {}
        query_inputs.update({
            f: 1 for f in fields_to_search
        })
        if score_modifiers:
            query_inputs.update(score_modifiers)

        query = {
            'yql': f'select {select_attributes} from {self._marqo_index.schema_name} where {search_term}',
            'model_restrict': self._marqo_index.schema_name,
            'hits': marqo_query.limit,
            'offset': marqo_query.offset,
            'query_features': query_inputs,
            'presentation.summary': summary,
            'ranking': ranking
        }
        query = {k: v for k, v in query.items() if v is not None}

        return query

    def _to_vespa_hybrid_query(self, marqo_query: MarqoHybridQuery) -> Dict[str, Any]:
        # Tensor term
        fields_to_search_tensor = self._get_tensor_fields_to_search(
            searchable_attributes=marqo_query.hybrid_parameters.searchableAttributesTensor
        )
        tensor_term = self._get_tensor_search_term(marqo_query) if fields_to_search_tensor else "False"

        # Lexical term
        fields_to_search_lexical = self._get_lexical_fields_to_search(
            searchable_attributes=marqo_query.hybrid_parameters.searchableAttributesLexical
        )
        lexical_term = self._get_lexical_search_term(marqo_query) if fields_to_search_lexical else "False"

        # If retrieval and ranking methods are opposite (lexical/tensor), use the rank() operator
        if (marqo_query.hybrid_parameters.retrievalMethod == RetrievalMethod.Lexical and
                marqo_query.hybrid_parameters.rankingMethod == RankingMethod.Tensor):
            individual_tensor_terms = self._get_individual_field_tensor_search_terms(marqo_query)
            lexical_term = f'rank({lexical_term}, {",".join(individual_tensor_terms)})'

        elif (marqo_query.hybrid_parameters.retrievalMethod == RetrievalMethod.Tensor and
              marqo_query.hybrid_parameters.rankingMethod == RankingMethod.Lexical):
            tensor_term = f'rank({tensor_term}, {lexical_term})'

        # Filter term
        filter_term = self._get_filter_term(marqo_query)
        if filter_term:
            filter_term = f' AND ({filter_term})'
        else:
            filter_term = ''
        select_attributes = self._get_select_attributes(marqo_query)
        summary = common.SUMMARY_ALL_VECTOR if marqo_query.expose_facets else common.SUMMARY_ALL_NON_VECTOR

        # Assign parameters to query
        query_inputs = {
            common.QUERY_INPUT_EMBEDDING: marqo_query.vector_query
        }

        # Separate fields to rank (lexical and tensor)
        query_inputs.update({
            common.QUERY_INPUT_HYBRID_FIELDS_TO_RANK_LEXICAL: {
                f: 1 for f in fields_to_search_lexical
            },
            common.QUERY_INPUT_HYBRID_FIELDS_TO_RANK_TENSOR: {
                f: 1 for f in fields_to_search_tensor
            }
        })

        """
        # TODO: implement this if no longer using custom searcher for lexical/tensor and tensor/lexical
        query_inputs.update({
            f: 1 for f in fields_to_search_lexical
        })
        query_inputs.update({
            f: 1 for f in fields_to_search_tensor
        })
        """

        # Extract score modifiers
        hybrid_score_modifiers = self._get_hybrid_score_modifiers(marqo_query)
        if hybrid_score_modifiers[constants.MARQO_SEARCH_METHOD_LEXICAL]:
            query_inputs.update(hybrid_score_modifiers[constants.MARQO_SEARCH_METHOD_LEXICAL])
        if hybrid_score_modifiers[constants.MARQO_SEARCH_METHOD_TENSOR]:
            query_inputs.update(hybrid_score_modifiers[constants.MARQO_SEARCH_METHOD_TENSOR])

        query = {
            'searchChain': 'marqo',
            'yql': 'PLACEHOLDER. WILL NOT BE USED IN HYBRID SEARCH.',
            'ranking': common.RANK_PROFILE_HYBRID_CUSTOM_SEARCHER,
            'ranking.rerankCount': marqo_query.limit + marqo_query.offset,       # limits the number of results going to phase 2
            
            'model_restrict': self._marqo_index.schema_name,
            'hits': marqo_query.limit,
            'offset': marqo_query.offset,
            'query_features': query_inputs,
            'presentation.summary': summary,

            # Custom searcher parameters
            'marqo__yql.tensor': f'select {select_attributes} from {self._marqo_index.schema_name} where {tensor_term}{filter_term}',
            'marqo__yql.lexical': f'select {select_attributes} from {self._marqo_index.schema_name} where ({lexical_term}){filter_term}',

            'marqo__ranking.lexical.lexical': common.RANK_PROFILE_BM25,
            'marqo__ranking.tensor.tensor': common.RANK_PROFILE_EMBEDDING_SIMILARITY,
            'marqo__ranking.lexical.tensor': common.RANK_PROFILE_HYBRID_BM25_THEN_EMBEDDING_SIMILARITY,
            'marqo__ranking.tensor.lexical': common.RANK_PROFILE_HYBRID_EMBEDDING_SIMILARITY_THEN_BM25,

            'marqo__hybrid.retrievalMethod': marqo_query.hybrid_parameters.retrievalMethod,
            'marqo__hybrid.rankingMethod': marqo_query.hybrid_parameters.rankingMethod,
            'marqo__hybrid.verbose': marqo_query.hybrid_parameters.verbose
        }

        query = {k: v for k, v in query.items() if v is not None}

        if marqo_query.hybrid_parameters.rankingMethod in {RankingMethod.RRF}: # TODO: Add NormalizeLinear
            query["marqo__hybrid.alpha"] = marqo_query.hybrid_parameters.alpha

        if marqo_query.hybrid_parameters.rankingMethod in {RankingMethod.RRF}:
            query["marqo__hybrid.rrf_k"] = marqo_query.hybrid_parameters.rrfK

        return query

    def _get_tensor_fields_to_search(
            self,
            marqo_query: Optional[MarqoTensorQuery] = None,
            searchable_attributes: Optional[List[str]] = None
    ) -> List[str]:
        if marqo_query is not None and searchable_attributes is not None:
            raise ValueError('Cannot provide both marqo_query and searchable_attributes')

        searchable_attributes = marqo_query.searchable_attributes if marqo_query is not None else searchable_attributes

        if searchable_attributes is not None:
            for att in searchable_attributes:
                if att not in self._marqo_index.tensor_field_map:
                    raise InvalidFieldNameError(
                        f'Index {self._marqo_index.name} has no tensor field {att}. '
                        f'Available tensor fields are: {", ".join(self._marqo_index.tensor_field_map.keys())}'
                    )

            fields_to_search = searchable_attributes
        else:
            fields_to_search = self._marqo_index.tensor_field_map.keys()

        if self._marqo_index_version < self._HYBRID_SEARCH_MINIMUM_VERSION:
            return fields_to_search
        else:
            return [self._marqo_index.tensor_field_map[f].embeddings_field_name
                    for f in fields_to_search]

    def _get_lexical_fields_to_search(
            self,
            marqo_query: Optional[MarqoLexicalQuery] = None,
            searchable_attributes: Optional[List[str]] = None
    ) -> List[str]:
        if marqo_query is not None and searchable_attributes is not None:
            raise ValueError('Cannot provide both marqo_query and searchable_attributes')

        searchable_attributes = marqo_query.searchable_attributes if marqo_query is not None else searchable_attributes

        if searchable_attributes is not None:
            for att in searchable_attributes:
                if att not in self._marqo_index.lexically_searchable_fields_names:
                    raise InvalidFieldNameError(
                        f'Index {self._marqo_index.name} has no lexically searchable field {att}. '
                        f'Available lexically searchable fields are: '
                        f'{", ".join(self._marqo_index.lexically_searchable_fields_names)}'
                    )

            fields_to_search = searchable_attributes
        else:
            fields_to_search = self._marqo_index.lexically_searchable_fields_names

        if self._marqo_index_version < self._HYBRID_SEARCH_MINIMUM_VERSION:
            return fields_to_search
        else:
            return [self._marqo_index.field_map[f].lexical_field_name
                    for f in fields_to_search]

    def _get_individual_field_tensor_search_terms(self, marqo_query: MarqoTensorQuery) -> List[str]:
        """
        Returns list of strings representing the tensor search terms for each field in the query.
        """
        if isinstance(marqo_query, MarqoHybridQuery):
            searchable_attributes = marqo_query.hybrid_parameters.searchableAttributesTensor
        else:
            searchable_attributes = marqo_query.searchable_attributes

        if searchable_attributes is not None:
            fields_to_search = [f for f in searchable_attributes if f in self._marqo_index.tensor_field_map]
        else:
            fields_to_search = self._marqo_index.tensor_field_map.keys()

        if marqo_query.ef_search is not None:
            target_hits = min(marqo_query.limit + marqo_query.offset, marqo_query.ef_search)
            additional_hits = max(marqo_query.ef_search - (marqo_query.limit + marqo_query.offset), 0)
        else:
            target_hits = marqo_query.limit + marqo_query.offset
            additional_hits = 0

        terms = []
        for field in fields_to_search:
            tensor_field = self._marqo_index.tensor_field_map[field]
            embedding_field_name = tensor_field.embeddings_field_name
            terms.append(
                f'('
                f'{{'
                f'targetHits:{target_hits}, '
                f'approximate:{str(marqo_query.approximate)}, '
                f'hnsw.exploreAdditionalHits:{additional_hits}'
                f'}}'
                f'nearestNeighbor({embedding_field_name}, {common.QUERY_INPUT_EMBEDDING})'
                f')'
            )
        return terms

    def _get_tensor_search_term(self, marqo_query: MarqoTensorQuery) -> str:
        terms = self._get_individual_field_tensor_search_terms(marqo_query)

        if terms:
            return f'({" OR ".join(terms)})'
        else:
            return ''

    def _get_filter_term(self, marqo_query: MarqoQuery) -> Optional[str]:
        # FIXME logic copied from unstructured
        def escape(s: str) -> str:
            return s.replace('\\', '\\\\').replace('"', '\\"')

        def generate_equality_filter_string(node: search_filter.EqualityTerm) -> str:
            filter_parts = []

            # Filter on `_id`
            if node.field == constants.MARQO_DOC_ID:
                return f'({unstructured_common.VESPA_FIELD_ID} contains "{escape(node.value)}")'

            # Bool Filter
            if node.value.lower() in self._FILTER_STRING_BOOL_VALUES:
                filter_value = int(True if node.value.lower() == "true" else False)
                bool_filter_string = (f'({unstructured_common.BOOL_FIELDS} contains '
                                      f'sameElement(key contains "{node.field}", value = {filter_value}))')
                filter_parts.append(bool_filter_string)

            # Short String Filter
            short_string_filter_string = (f'({unstructured_common.SHORT_STRINGS_FIELDS} '
                                          f'contains sameElement(key contains "{node.field}", '
                                          f'value contains "{escape(node.value)}"))')
            filter_parts.append(short_string_filter_string)

            # String Array Filter
            string_array_filter_string = (f'({unstructured_common.STRING_ARRAY} contains '
                                          f'"{node.field}::{escape(node.value)}")')
            filter_parts.append(string_array_filter_string)

            # Numeric Filter
            numeric_filter_string = ""
            try:
                numeric_value = int(node.value)
                numeric_filter_string = (
                    f'({unstructured_common.INT_FIELDS} contains sameElement(key contains "{node.field}", value = {numeric_value})) '
                    f'OR ({unstructured_common.FLOAT_FIELDS} contains sameElement(key contains "{node.field}", value = {numeric_value}))')
            except ValueError:
                try:
                    numeric_value = float(node.value)
                    numeric_filter_string = f'({unstructured_common.FLOAT_FIELDS} contains sameElement(key contains "{node.field}", value = {numeric_value}))'
                except ValueError:
                    pass

            if numeric_filter_string:
                filter_parts.append(numeric_filter_string)

            # Final Filter String
            final_filter_string = f"({' OR '.join(filter_parts)})"
            return final_filter_string

        def generate_range_filter_string(node: search_filter.RangeTerm) -> str:
            lower = f'value >= {node.lower}' if node.lower is not None else ""
            higher = f'value <= {node.upper}' if node.upper is not None else ""
            bound = f'{lower}, {higher}' if lower and higher else f'{lower}{higher}'
            if not bound:
                raise InternalError('RangeTerm has no lower or upper bound')

            float_field_string = (f'({unstructured_common.FLOAT_FIELDS} contains '
                                  f'sameElement(key contains "{node.field}", {bound}))')

            int_field_string = (f'({unstructured_common.INT_FIELDS} contains '
                                f'sameElement(key contains "{node.field}", {bound}))')

            return f'({float_field_string} OR {int_field_string})'

        def tree_to_filter_string(node: search_filter.Node) -> str:
            if isinstance(node, search_filter.Operator):
                if isinstance(node, search_filter.And):
                    operator = 'AND'
                elif isinstance(node, search_filter.Or):
                    operator = 'OR'
                else:
                    raise InternalError(f'Unknown operator type {type(node)}')
                return f'({tree_to_filter_string(node.left)} {operator} {tree_to_filter_string(node.right)})'
            elif isinstance(node, search_filter.Modifier):
                if isinstance(node, search_filter.Not):
                    return f'!({tree_to_filter_string(node.modified)})'
                else:
                    raise InternalError(f'Unknown modifier type {type(node)}')
            elif isinstance(node, search_filter.Term):
                if isinstance(node, search_filter.EqualityTerm):
                    return generate_equality_filter_string(node)
                elif isinstance(node, search_filter.RangeTerm):
                    return generate_range_filter_string(node)
                elif isinstance(node, search_filter.InTerm):
                    raise InvalidArgumentError("The 'IN' filter keyword is not yet supported for unstructured indexes")
            raise InternalError(f'Unknown node type {type(node)}')

        if marqo_query.filter is not None:
            return tree_to_filter_string(marqo_query.filter.root)

    def _get_select_attributes(self, marqo_query: MarqoQuery) -> str:
        if marqo_query.attributes_to_retrieve is not None:
            return ', '.join(marqo_query.attributes_to_retrieve)
        else:
            return '*'

    def _get_lexical_search_term(self, marqo_query: MarqoLexicalQuery) -> str:
        if isinstance(marqo_query, MarqoHybridQuery):
            score_modifiers = marqo_query.hybrid_parameters.scoreModifiersLexical
        else:
            score_modifiers = marqo_query.score_modifiers

        # Empty query and wildcard
        if not marqo_query.or_phrases and not marqo_query.and_phrases:
            return 'false'
        if marqo_query.or_phrases == ["*"] and not marqo_query.and_phrases:
            return 'true'

        # Optional tokens
        if marqo_query.or_phrases and score_modifiers:
            or_terms = ' OR '.join([
                self._get_lexical_contains_term(phrase, marqo_query) for phrase in marqo_query.or_phrases
            ])
        elif marqo_query.or_phrases and not score_modifiers:
            or_terms = 'weakAnd(%s)' % ', '.join([
                self._get_lexical_contains_term(phrase, marqo_query) for phrase in marqo_query.or_phrases
            ])
        else:
            or_terms = ''

        # Required tokens
        if marqo_query.and_phrases:
            and_terms = ' AND '.join([
                self._get_lexical_contains_term(phrase, marqo_query) for phrase in marqo_query.and_phrases
            ])
            if or_terms:
                or_terms = f'({or_terms})'
                and_terms = f' AND ({and_terms})'
        else:
            and_terms = ''

        return f'{or_terms}{and_terms}'

    def _get_lexical_contains_term(self, phrase, query: MarqoQuery) -> str:
        if isinstance(query, MarqoHybridQuery):
            searchable_attributes = query.hybrid_parameters.searchableAttributesLexical
        else:
            searchable_attributes = query.searchable_attributes

        if searchable_attributes is not None:
            return ' OR '.join([
                f'{self._marqo_index.field_map[field].lexical_field_name} contains "{phrase}"'
                for field in searchable_attributes
            ])
        else:
            return f'default contains "{phrase}"'

    def _verify_marqo_field_name(self, field_name: str):
        field_map = self._marqo_index.field_map
        if field_name not in field_map:
            raise InvalidFieldNameError(f'Invalid field name {field_name} for index {self._marqo_index.name}. '
                                        f'Valid field names are {", ".join(field_map.keys())}')

    def _verify_marqo_tensor_field_name(self, field_name: str):
        tensor_field_map = self._marqo_index.tensor_field_map
        if field_name not in tensor_field_map:
            raise InvalidFieldNameError(f'Invalid tensor field name {field_name} for index {self._marqo_index.name}. '
                                        f'Valid tensor field names are {", ".join(tensor_field_map.keys())}')

    def _verify_marqo_tensor_field(self, field_name: str, field_value: Dict[str, Any]):
        if not set(field_value.keys()) == {constants.MARQO_DOC_CHUNKS, constants.MARQO_DOC_EMBEDDINGS}:
            raise InternalError(f'Invalid tensor field {field_name}. '
                                f'Expected keys {constants.MARQO_DOC_CHUNKS}, {constants.MARQO_DOC_EMBEDDINGS} '
                                f'but found {", ".join(field_value.keys())}')

    def _verify_marqo_field_type(self, field_name: str, value: Any):
        marqo_type = self._marqo_index.field_map[field_name].type
        python_type = self._get_python_type(marqo_type)

        if isinstance(python_type, tuple):
            # Logic branch for array types
            if ((not isinstance(value, python_type[0])) or
                    (isinstance(value, list) and not all(isinstance(v, python_type[1]) for v in value))):
                raise InvalidDataTypeError(f'Invalid value {value} for a list field {field_name} with Marqo type '
                                           f'{marqo_type.value}. All list elements must be the same valid type ')

        elif (
                (isinstance(python_type, list) and not any(isinstance(value, t) for t in python_type)) or
                (not isinstance(python_type, list) and not isinstance(value, python_type))
        ):
            raise InvalidDataTypeError(f'Invalid value {value} for field {field_name} with Marqo type '
                                       f'{marqo_type.value}. Expected a value of type {python_type}, but found '
                                       f'{type(value)}')
        else:
            ValueError(f'Invalid python type {python_type} for field {field_name} with Marqo type {marqo_type.value} '
                       f'during call to _verify_marqo_field_type')

    def _verify_numerical_field_value(self, value: Union[float, int], index_field: Field):
        if index_field.type in (FieldType.Float, FieldType.ArrayFloat, FieldType.MapFloat):
            self._verify_float_field_range(value)
        elif index_field.type in (FieldType.Int, FieldType.ArrayInt, FieldType.MapInt):
            self._verify_int_field_range(value)
        elif index_field.type in (FieldType.Long, FieldType.ArrayLong, FieldType.MapLong):
            self._verify_long_field_range(value)
        elif index_field.type in (FieldType.Double, FieldType.ArrayDouble, FieldType.MapDouble):
            pass
        else:
            raise InternalError(f'Invalid field type {index_field.type} for field {index_field.name} called by'
                                f'_verify_numerical_field_value. Expected one of {FieldType.Float}, {FieldType.Int}, '
                                f'{FieldType.Long}, {FieldType.Double}')

    def _verify_float_field_range(self, value: float):
        if not (self._MIN_FLOAT <= value <= self._MAX_FLOAT):
            raise InvalidDataRangeError(f'Invalid value {value} for float field. Expected a value in the range '
                                        f'[{self._MIN_FLOAT}, {self._MAX_FLOAT}], but found {value}. '
                                        f'If you wish to store a value outside of this range, create a field with type '
                                        f"'{FieldType.Double}' ")

    def _verify_int_field_range(self, value: int):
        if not (self._MIN_INT <= value <= self._MAX_INT):
            raise InvalidDataRangeError(f"Invalid value {value} for int field. Expected a value in the range "
                                        f"[{self._MIN_INT}, {self._MAX_INT}], but found {value}. "
                                        f"If you wish to store a value outside of this range, create a field with type "
                                        f"'{FieldType.Long} or '{FieldType.Double}' ")

    def _verify_long_field_range(self, value: int):
        if not (self._MIN_LONG <= value <= self._MAX_LONG):
            raise InvalidDataRangeError(f"Invalid value {value} for long field. Expected a value in the range "
                                        f"[{self._MIN_LONG}, {self._MAX_LONG}], but found {value}. "
                                        f"If you wish to store a value outside of this range, create a field with type "
                                        f"'{FieldType.Double}' ")

    def _verify_id_field(self, value: str):
        """Validates that the _id value is acceptable.

        Args:
            value: The _id value to validate
        """
        if not isinstance(value, str):
            raise MarqoDocumentParsingError(
                "Document _id must be a string type! "
                f"Received _id {value} of type `{type(value).__name__}`")
        if not value:
            raise MarqoDocumentParsingError("Document ID can't be empty")

    def _get_python_type(self, marqo_type: FieldType) -> type:
        try:
            return self._MARQO_TO_PYTHON_TYPE_MAP[marqo_type]
        except KeyError:
            raise InternalError(f'Unknown Marqo type: {marqo_type}')
