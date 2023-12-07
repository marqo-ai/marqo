import json
import struct
import base64
from copy import deepcopy
from typing import List, Dict, Any

from pydantic import Field, BaseModel

from marqo.core import constants as index_constants
from marqo.core.exceptions import VespaDocumentParsingError
from marqo.core.unstructured_vespa_index import common as unstructured_common
from marqo.core.unstructured_vespa_index import constants as unstructured_constants


class UnstructuredVespaDocumentFields(BaseModel):
    """A class with fields that are common to all Vespa documents."""
    marqo__id: str = Field(alias=unstructured_constants.VESPA_FIELD_ID)

    strings: List[str] = Field(default_factory=list, alias=unstructured_common.STRINGS)
    long_string_fields: Dict[str, str] = Field(default_factory=dict, alias=unstructured_common.LONGS_STRINGS_FIELDS)
    short_string_fields: Dict[str, str] = Field(default_factory=dict, alias=unstructured_common.SHORT_STRINGS_FIELDS)
    string_arrays: List[str] = Field(default_factory=list, alias=unstructured_common.STRING_ARRAY)
    int_fields: Dict[str, int] = Field(default_factory=dict, alias=unstructured_common.INT_FIELDS)
    bool_fields: Dict[str, bool] = Field(default_factory=dict, alias=unstructured_common.BOOL_FIELDS)
    float_fields: Dict[str, float] = Field(default_factory=dict, alias=unstructured_common.FLOAT_FIELDS)
    score_modifiers_fields: Dict[str, Any] = Field(default_factory=dict, alias=unstructured_common.SCORE_MODIFIERS)
    vespa_chunks: List[str] = Field(default_factory=list, alias=unstructured_common.VESPA_DOC_CHUNKS)
    vespa_embeddings: Dict[str, Any] = Field(default_factory=dict, alias=unstructured_common.VESPA_DOC_EMBEDDINGS)
    vespa_multimodal_params: Dict[str, str] = Field(default_factory=str,
                                                    alias=unstructured_common.VESPA_DOC_MULTIMODAL_PARAMS)
    vector_counts: int = Field(default=0, alias=unstructured_common.FIELD_VECTOR_COUNT)

    match_features: Dict[str, Any] = Field(default_factory=dict, alias=unstructured_constants.VESPA_DOC_MATCH_FEATURES)

    # Fields that are excluded when generating vespa documents
    _EXCLUDED_FIELDS = {"match_features"}

    class Config:
        allow_population_by_field_name = True

    def to_vespa_dictionary(self) -> Dict[str, Any]:
        # Exclude None and empty fields but keep 0 as we need to have vector_counts in docs
        return {k: v for k, v in self.dict(exclude_none=True, by_alias=True,
                                           exclude=self._EXCLUDED_FIELDS).items() if v or v == 0}

    def to_marqo_doc(self, return_highlights: bool = False) -> Dict[str, Any]:
        marqo_document = {}
        # Processing short_string_fields and long_string_fields back into original format
        marqo_document.update(self.short_string_fields)
        marqo_document.update(self.long_string_fields)
        # Reconstruct string arrays
        for string_array in self.string_arrays:
            key, value = string_array.split("::", 2)
            if key not in marqo_document:
                marqo_document[key] = []
            marqo_document[key].append(value)

        # Add int and float fields back
        marqo_document.update(self.int_fields)
        marqo_document.update(self.float_fields)
        marqo_document.update({k: bool(v) for k, v in self.bool_fields.items()})
        marqo_document[index_constants.MARQO_DOC_ID] = self.marqo__id

        # vespa_embeddings may not return if it comes from a search result
        embeddings_list = []
        if self.vespa_embeddings:
            try:
                embeddings_list = list(self.vespa_embeddings["blocks"].values())
            except (KeyError):
                raise VespaDocumentParsingError(f"No embeddings found in the document _id = {self.marqo__id}")

        if self.vespa_chunks and embeddings_list:
            if not len(self.vespa_chunks) == len(embeddings_list):
                raise ValueError(f"Number of chunks and embeddings do not match for document _id= {self.marqo__id}")
            marqo_document[index_constants.MARQO_DOC_TENSORS] = dict()
            for chunk, embedding in zip(self.vespa_chunks, embeddings_list):
                if "::" not in chunk:
                    raise ValueError(f"Chunk {chunk} does not have a field_name::content format")
                field_name, content = chunk.split("::", 2)
                if field_name not in marqo_document[index_constants.MARQO_DOC_TENSORS]:
                    marqo_document[index_constants.MARQO_DOC_TENSORS][field_name] = dict()
                    marqo_document[index_constants.MARQO_DOC_TENSORS][field_name][index_constants.MARQO_DOC_CHUNKS] = []
                    marqo_document[index_constants.MARQO_DOC_TENSORS][field_name][
                        index_constants.MARQO_DOC_EMBEDDINGS] = []
                marqo_document[index_constants.MARQO_DOC_TENSORS][field_name][index_constants.MARQO_DOC_CHUNKS].append(
                    content)
                marqo_document[index_constants.MARQO_DOC_TENSORS][field_name][
                    index_constants.MARQO_DOC_EMBEDDINGS].append(embedding)

        if self.vespa_multimodal_params:
            marqo_document[unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS] = dict()
            for multimodal_field_name, serialized_multimodal_params in self.vespa_multimodal_params.items():
                marqo_document[unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS][multimodal_field_name] = \
                    json.loads(serialized_multimodal_params)

        if return_highlights and self.match_features:
            marqo_document[index_constants.MARQO_DOC_HIGHLIGHTS] = self._extract_highlights()

        return marqo_document

    def _extract_highlights(self):
        if not self.match_features:
            raise ValueError("No match features found in the document")
        try:
            chunk_index: int = int(list(self.match_features[f"closest({unstructured_common.VESPA_DOC_EMBEDDINGS})"] \
                                            ["cells"].keys())[0])
        except KeyError:
            raise ValueError("No match features found in the document")
        field_name, content = self.vespa_chunks[chunk_index].split("::", 2)
        return {field_name: content}


class UnstructuredIndexDocument(BaseModel):
    """A helper class to handle the conversion between Vespa and Marqo documents for an unstructured index.

    The object can be instantiated from a Marqo document using the from_marqo_document method,
    or can be instantiated from a Vespa document using the from_vespa_document method.
    """
    id: str = Field(alias=index_constants.VESPA_DOC_ID)
    fields: UnstructuredVespaDocumentFields = Field(default_factory=UnstructuredVespaDocumentFields,
                                                    alias=index_constants.VESPA_DOC_FIELDS)

    class Config:
        allow_population_by_field_name = True

    @classmethod
    def from_vespa_document(cls, document: Dict) -> "UnstructuredIndexDocument":
        """Instantiate an UnstructuredIndexDocument from a Vespa document."""
        fields = document.get(index_constants.VESPA_DOC_FIELDS, {})
        return cls(id=document[index_constants.VESPA_DOC_ID],
                   fields=UnstructuredVespaDocumentFields(**fields))

    @classmethod
    def from_marqo_document(cls, document: Dict) -> "UnstructuredIndexDocument":
        """Instantiate an UnstructuredIndexDocument from a valid Marqo document from
        add_documents"""

        if index_constants.MARQO_DOC_ID not in document:
            raise ValueError(f"Unstructured Marqo document does not have a {index_constants.MARQO_DOC_ID} field. "
                             f"This should be assigned for a valid document")

        copied = deepcopy(document)
        doc_id = copied[index_constants.MARQO_DOC_ID]
        instance = cls(id=doc_id, fields=UnstructuredVespaDocumentFields(marqo__id=doc_id))
        del copied[index_constants.MARQO_DOC_ID]

        for key, value in copied.items():
            if key in [index_constants.MARQO_DOC_EMBEDDINGS, index_constants.MARQO_DOC_CHUNKS,
                       unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS]:
                continue
            if isinstance(value, str):
                if len(value) <= unstructured_constants.SHORT_STRING_THRESHOLD:
                    instance.fields.short_string_fields[key] = value
                else:
                    instance.fields.long_string_fields[key] = value
                instance.fields.strings.append(value)
            elif isinstance(value, bool):
                instance.fields.bool_fields[key] = int(value)
            elif isinstance(value, list) and all(isinstance(elem, str) for elem in value):
                instance.fields.string_arrays.extend([f"{key}::{element}" for element in value])
            elif isinstance(value, int):
                instance.fields.int_fields[key] = value
                instance.fields.score_modifiers_fields[key] = value
            elif isinstance(value, float):
                instance.fields.float_fields[key] = value
                instance.fields.score_modifiers_fields[key] = value

        instance.fields.vespa_multimodal_params = copied.get(unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS, {})
        instance.fields.vespa_embeddings = copied.get(index_constants.MARQO_DOC_EMBEDDINGS, {})
        instance.fields.vespa_chunks = copied.get(index_constants.MARQO_DOC_CHUNKS, [])
        instance.fields.vector_counts = len(instance.fields.vespa_embeddings)
        return instance

    def to_vespa_document(self) -> Dict[str, Any]:
        """Convert VespaDocumentObject to a Vespa document.
        Empty fields are removed from the document."""
        return {index_constants.VESPA_DOC_ID: self.id,
                index_constants.VESPA_DOC_FIELDS: self.fields.to_vespa_dictionary()}

    def to_marqo_document(self, return_highlights: bool = False) -> Dict[str, Any]:
        """Convert VespaDocumentObject back to the original document structure."""
        return self.fields.to_marqo_doc(return_highlights=return_highlights)
