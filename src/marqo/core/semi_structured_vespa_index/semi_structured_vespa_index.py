from typing import Dict, Any, Optional, cast

from marqo.core.constants import MARQO_DOC_HIGHLIGHTS, MARQO_DOC_ID
from marqo.core.exceptions import MarqoDocumentParsingError
from marqo.core.models import MarqoQuery
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex
from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoLexicalQuery, MarqoHybridQuery
from marqo.core.semi_structured_vespa_index import common
from marqo.core.semi_structured_vespa_index.semi_structured_document import SemiStructuredVespaDocument
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
from marqo.core.unstructured_vespa_index.unstructured_vespa_index import UnstructuredVespaIndex
from marqo.exceptions import InternalError


class SemiStructuredVespaIndex(StructuredVespaIndex, UnstructuredVespaIndex):
    """
    An implementation of VespaIndex for SemiStructured indexes.
    """

    def __init__(self, marqo_index: SemiStructuredMarqoIndex):
        super().__init__(marqo_index)

    def get_marqo_index(self) -> SemiStructuredMarqoIndex:
        if isinstance(self._marqo_index, SemiStructuredMarqoIndex):
            return cast(SemiStructuredMarqoIndex, self._marqo_index)
        else:
            raise TypeError('Wrong type of marqo index')

    def to_vespa_document(self, marqo_document: Dict[str, Any]) -> Dict[str, Any]:
        return (SemiStructuredVespaDocument.from_marqo_document(
            marqo_document, marqo_index=self.get_marqo_index())).to_vespa_document()

    def to_marqo_document(self, vespa_document: Dict[str, Any], return_highlights: bool = False) -> Dict[str, Any]:
        vespa_doc = SemiStructuredVespaDocument.from_vespa_document(vespa_document, marqo_index=self.get_marqo_index())
        marqo_doc = vespa_doc.to_marqo_document(marqo_index=self.get_marqo_index())

        if return_highlights and vespa_doc.match_features:
            # Since tensor fields are stored in each individual field, we need to use same logic in structured
            # index to extract highlights
            marqo_doc[MARQO_DOC_HIGHLIGHTS] = StructuredVespaIndex._extract_highlights(
                self, vespa_document.get('fields', {}))

        return marqo_doc

    def to_vespa_query(self, marqo_query: MarqoQuery) -> Dict[str, Any]:
        # Verify attributes to retrieve, if defined
        if marqo_query.attributes_to_retrieve is not None:
            marqo_query.attributes_to_retrieve.append(common.VESPA_FIELD_ID)
            # add chunk field names for tensor fields
            marqo_query.attributes_to_retrieve.extend(
                [self.get_marqo_index().tensor_field_map[att].chunk_field_name
                 for att in marqo_query.attributes_to_retrieve
                 if att in self.get_marqo_index().tensor_field_map]
            )

        # Hybrid must be checked first since it is a subclass of Tensor and Lexical
        if isinstance(marqo_query, MarqoHybridQuery):
            return StructuredVespaIndex._to_vespa_hybrid_query(self, marqo_query)
        elif isinstance(marqo_query, MarqoTensorQuery):
            return StructuredVespaIndex._to_vespa_tensor_query(self, marqo_query)
        elif isinstance(marqo_query, MarqoLexicalQuery):
            return StructuredVespaIndex._to_vespa_lexical_query(self, marqo_query)

        else:
            raise InternalError(f'Unknown query type {type(marqo_query)}')

    @classmethod
    def _get_filter_term(cls, marqo_query: MarqoQuery) -> Optional[str]:
        # Reuse logic in UnstructuredVespaIndex to create filter term
        return UnstructuredVespaIndex._get_filter_term(marqo_query)

    def to_vespa_partial_document(self, marqo_document: Dict[str, Any],
                                  original_marqo_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        vespa_id: Optional[str] = None
        vespa_fields: Dict[str, Any] = dict()
        int_fields_changed = False
        float_fields_changed = False

        if MARQO_DOC_ID not in marqo_document:
            raise MarqoDocumentParsingError(f"'{MARQO_DOC_ID}' is a required field but it does not exist")
        else:
            vespa_id = marqo_document[MARQO_DOC_ID]
            self._verify_id_field(vespa_id)

        for marqo_field in marqo_document:
            if marqo_field == MARQO_DOC_ID:
                continue

            # TODO error out if marqo_field is tensor field

            if marqo_field in original_marqo_document:
                if type(marqo_document[marqo_field]) != type(original_marqo_document[marqo_field]):
                    raise MarqoDocumentParsingError(f"'{marqo_field}' type mismatch, expected "
                                                    f"{type(original_marqo_document[marqo_field])}, "
                                                    f"but was {type(marqo_document[marqo_field])}")
                if marqo_document[marqo_field] == original_marqo_document[marqo_field]:
                    continue

            if isinstance(marqo_document[marqo_field], int):
                field_name = f'{common.INT_FIELDS}{{{marqo_field}}}'
                vespa_fields[field_name] = {"assign": marqo_document[marqo_field]}
                int_fields_changed = True

            if isinstance(marqo_document[marqo_field], float):
                field_name = f'{common.FLOAT_FIELDS}{{{marqo_field}}}'
                vespa_fields[field_name] = {"assign": marqo_document[marqo_field]}
                float_fields_changed = True

            original_marqo_document[marqo_field] = marqo_document[marqo_field]

        doc = SemiStructuredVespaDocument.from_marqo_document(original_marqo_document, self.get_marqo_index())
        if int_fields_changed or float_fields_changed:
            vespa_fields[common.SCORE_MODIFIERS] = {
                "modify": {
                    "operation": "replace",
                    "cells": doc.fixed_fields.score_modifiers_fields
                }
            }

        return {"id": vespa_id, "create_timestamp": doc.fixed_fields.create_timestamp,  "fields": vespa_fields}


