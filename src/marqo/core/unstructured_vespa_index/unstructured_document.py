import json
from copy import deepcopy
from typing import List, Dict, Any

from pydantic import Field, BaseModel
from marqo.base_model import MarqoBaseModel

from marqo.core import constants as index_constants
from marqo.core.exceptions import VespaDocumentParsingError
from marqo.core.unstructured_vespa_index import common as unstructured_common


class UnstructuredVespaDocumentFields(MarqoBaseModel):
    """A class with fields that are common to all Vespa documents."""
    marqo__id: str = Field(alias=unstructured_common.VESPA_FIELD_ID)

    strings: List[str] = Field(default_factory=list, alias=unstructured_common.STRINGS)
    long_string_fields: Dict[str, str] = Field(default_factory=dict, alias=unstructured_common.LONGS_STRINGS_FIELDS)
    short_string_fields: Dict[str, str] = Field(default_factory=dict, alias=unstructured_common.SHORT_STRINGS_FIELDS)
    string_arrays: List[str] = Field(default_factory=list, alias=unstructured_common.STRING_ARRAY)
    int_fields: Dict[str, int] = Field(default_factory=dict, alias=unstructured_common.INT_FIELDS)
    bool_fields: Dict[str, int] = Field(default_factory=dict, alias=unstructured_common.BOOL_FIELDS)
    float_fields: Dict[str, float] = Field(default_factory=dict, alias=unstructured_common.FLOAT_FIELDS)
    score_modifiers_fields: Dict[str, Any] = Field(default_factory=dict, alias=unstructured_common.SCORE_MODIFIERS)
    vespa_chunks: List[str] = Field(default_factory=list, alias=unstructured_common.VESPA_DOC_CHUNKS)
    vespa_embeddings: Dict[str, Any] = Field(default_factory=dict, alias=unstructured_common.VESPA_DOC_EMBEDDINGS)
    vespa_multimodal_params: Dict[str, str] = Field(default_factory=str,
                                                    alias=unstructured_common.VESPA_DOC_MULTIMODAL_PARAMS)
    vector_counts: int = Field(default=0, alias=unstructured_common.FIELD_VECTOR_COUNT)

    match_features: Dict[str, Any] = Field(default_factory=dict, alias=unstructured_common.VESPA_DOC_MATCH_FEATURES)

    def extract_highlights(self) -> Dict[str, Any]:
        if not self.match_features:
            raise VespaDocumentParsingError("No match features found in the document")
        try:
            chunk_index: int = int(list(self.match_features[f"closest({unstructured_common.VESPA_DOC_EMBEDDINGS})"] \
                                            ["cells"].keys())[0])
        except KeyError:
            raise VespaDocumentParsingError("No match features found in the document")
        field_name, content = self.vespa_chunks[chunk_index].split("::", 2)
        return {field_name: content}


class UnstructuredVespaDocument(MarqoBaseModel):
    """A helper class to handle the conversion between Vespa and Marqo documents for an unstructured index.

    The object can be instantiated from a Marqo document using the from_marqo_document method,
    or can be instantiated from a Vespa document using the from_vespa_document method.
    """
    id: str
    fields: UnstructuredVespaDocumentFields = Field(default_factory=UnstructuredVespaDocumentFields)

    _VESPA_DOC_FIELDS = "fields"
    _VESPA_DOC_ID = "id"

    @classmethod
    def from_vespa_document(cls, document: Dict) -> "UnstructuredVespaDocument":
        """Instantiate an UnstructuredVespaDocument from a Vespa document."""
        fields = document.get(cls._VESPA_DOC_FIELDS, {})
        return cls(id=document[cls._VESPA_DOC_ID],
                   fields=UnstructuredVespaDocumentFields(**fields))

    @classmethod
    def from_marqo_document(cls, document: Dict, filter_string_max_length: int) -> "UnstructuredVespaDocument":
        """Instantiate an UnstructuredVespaDocument from a valid Marqo document from
        add_documents"""

        if index_constants.MARQO_DOC_ID not in document:
            raise VespaDocumentParsingError(f"Unstructured Marqo document does not have a {index_constants.MARQO_DOC_ID} field. "
                             f"This should be assigned for a valid document")

        doc_id = document[index_constants.MARQO_DOC_ID]
        instance = cls(id=doc_id, fields=UnstructuredVespaDocumentFields(marqo__id=doc_id))

        for key, value in document.items():
            if key in [index_constants.MARQO_DOC_EMBEDDINGS, index_constants.MARQO_DOC_CHUNKS,
                       unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS, index_constants.MARQO_DOC_ID]:
                continue
            if isinstance(value, str):
                if len(value) <= filter_string_max_length:
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
            else:
                raise VespaDocumentParsingError(f"Document {document} with field {key} has an "
                                 f"unsupported type {type(value)} which has not been validated in advance.")

        instance.fields.vespa_multimodal_params = document.get(unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS, {})
        instance.fields.vespa_embeddings = document.get(index_constants.MARQO_DOC_EMBEDDINGS, {})
        instance.fields.vespa_chunks = document.get(index_constants.MARQO_DOC_CHUNKS, [])
        instance.fields.vector_counts = len(instance.fields.vespa_embeddings)
        return instance

    def to_vespa_document(self) -> Dict[str, Any]:
        """Convert VespaDocumentObject to a Vespa document.
        Empty fields are removed from the document."""
        vespa_fields = {k: v for k, v in self.fields.dict(exclude_none=True, by_alias=True).items() if v or v == 0}

        return {self._VESPA_DOC_ID: self.id, self._VESPA_DOC_FIELDS: vespa_fields}

    def to_marqo_document(self, return_highlights: bool = False) -> Dict[str, Any]:
        """Convert VespaDocumentObject back to the original document structure."""
        marqo_document = {}
        # Processing short_string_fields and long_string_fields back into original format
        marqo_document.update(self.fields.short_string_fields)
        marqo_document.update(self.fields.long_string_fields)
        # Reconstruct string arrays
        for string_array in self.fields.string_arrays:
            key, value = string_array.split("::", 2)
            if key not in marqo_document:
                marqo_document[key] = []
            marqo_document[key].append(value)

        # Add int and float fields back
        marqo_document.update(self.fields.int_fields)
        marqo_document.update(self.fields.float_fields)
        marqo_document.update({k: bool(v) for k, v in self.fields.bool_fields.items()})
        marqo_document[index_constants.MARQO_DOC_ID] = self.fields.marqo__id

        # This normally means the document is return with show_vectors=True
        if self.fields.vespa_chunks and self.fields.vespa_embeddings:
            try:
                embeddings_list = list(self.fields.vespa_embeddings["blocks"].values())
            except (KeyError, AttributeError, TypeError) as e:
                raise VespaDocumentParsingError(f"Can not parsing embeddings for document "
                                                f"_id={self.fields.marqo__id}. "
                                                f"Document={self.fields.dict()}. "
                                                f"Original error message {e}")

            if not len(self.fields.vespa_chunks) == len(embeddings_list):
                raise VespaDocumentParsingError(f"Number of chunks and embeddings do not match "
                                                f"for document _id= {self.fields.marqo__id}")
            marqo_document[index_constants.MARQO_DOC_TENSORS] = dict()
            for chunk, embedding in zip(self.fields.vespa_chunks, embeddings_list):
                if "::" not in chunk:
                    raise VespaDocumentParsingError(f"Chunk {chunk} does not have a field_name::content format")
                field_name, content = chunk.split("::", 2)
                if field_name not in marqo_document[index_constants.MARQO_DOC_TENSORS]:
                    marqo_document[index_constants.MARQO_DOC_TENSORS][field_name] = dict()
                    marqo_document[index_constants.MARQO_DOC_TENSORS][field_name][index_constants.MARQO_DOC_CHUNKS]\
                        = []
                    marqo_document[index_constants.MARQO_DOC_TENSORS][field_name][
                        index_constants.MARQO_DOC_EMBEDDINGS] = []
                marqo_document[index_constants.MARQO_DOC_TENSORS][field_name] \
                    [index_constants.MARQO_DOC_CHUNKS].append(
                    content)
                marqo_document[index_constants.MARQO_DOC_TENSORS][field_name][
                    index_constants.MARQO_DOC_EMBEDDINGS].append(embedding)

        if self.fields.vespa_multimodal_params:
            marqo_document[unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS] = dict()
            for multimodal_field_name, serialized_multimodal_params in self.fields.vespa_multimodal_params.items():
                marqo_document[unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS][multimodal_field_name] = \
                    json.loads(serialized_multimodal_params)

        if return_highlights and self.fields.match_features:
            if not self.fields.vespa_chunks:
                raise VespaDocumentParsingError(f"Document {self.fields.marqo__id} does not have any chunks. "
                                                f"No highlights can be extracted.")
            marqo_document[index_constants.MARQO_DOC_HIGHLIGHTS] = self.fields.extract_highlights()

        return marqo_document