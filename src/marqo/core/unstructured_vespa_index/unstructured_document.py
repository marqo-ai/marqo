from typing import List, Dict, Any, Union
from copy import deepcopy

from pydantic import Field, BaseModel
from marqo.core import constants as index_constants
from marqo.core.unstructured_vespa_index import common as unstructured_common
from marqo.core.unstructured_vespa_index import constants as unstructured_constants


class UnstructuredVespaDocumentFields(BaseModel):
    """A class with fields that are common to all Vespa documents."""
    marqo__id: str = Field(default_factory=str, alias=index_constants.VESPA_FIELD_ID)
    strings: List[str] = Field(default_factory=list, alias=unstructured_common.STRINGS)
    long_string_fields: Dict[str, str] = Field(default_factory=dict, alias=unstructured_common.LONGS_STRINGS_FIELDS)
    short_string_fields: Dict[str, str] = Field(default_factory=dict, alias=unstructured_common.SHORT_STRINGS_FIELDS)
    string_arrays: List[str] = Field(default_factory=list, alias=unstructured_common.STRING_ARRAY)
    int_fields: Dict[str, int] = Field(default_factory=dict, alias=unstructured_common.INT_FIELDS)
    float_fields: Dict[str, float] = Field(default_factory=dict, alias=unstructured_common.FLOAT_FIELDS)
    score_modifiers_fields: Dict[str, Union[float,int]] = Field(default_factory=dict, alias=unstructured_common.SCORE_MODIFIERS)
    vespa_chunks: List[str] = Field(default_factory=list, alias=unstructured_common.VESPA_DOC_CHUNKS)
    vespa_embeddings: Dict = Field(default_factory=dict, alias=unstructured_common.VESPA_DOC_EMBEDDINGS)
    match_features: Dict[str, Any] = Field(default_factory=dict, alias=index_constants.VESPA_DOC_MATCH_FEATURES)

    class Config:
        allow_population_by_field_name = True

    def to_vespa_dictionary(self) -> Dict[str, Any]:
        # Exclude None and empty fields
        return {
            key: value for key, value in self.dict(exclude_none=True, by_alias=True).items()
            if value
        }

    def to_marqo_doc(self, return_highlights: bool = False) -> Dict[str, Any]:
        marqo_document = {}
        # Processing short_string_fields and long_string_fields back into original format
        marqo_document.update(self.short_string_fields)
        marqo_document.update(self.long_string_fields)
        # Reconstruct string arrays
        for string_array in self.string_arrays:
            key, value = string_array.split("::", 1)
            if key not in marqo_document:
                marqo_document[key] = []
            marqo_document[key].append(value)

        # Add int and float fields back
        marqo_document.update(self.int_fields)
        marqo_document.update(self.float_fields)
        marqo_document[index_constants.MARQO_DOC_ID] = self.marqo__id

        if self.vespa_chunks and self.vespa_embeddings:
            if not len(self.vespa_chunks) == len(self.vespa_embeddings):
                raise ValueError(f"Number of chunks and embeddings do not match for document _id= {self.marqo__id}")
            marqo_document[index_constants.MARQO_DOC_TENSORS] = dict()
            for chunk, embedding in zip(self.vespa_chunks, self.vespa_embeddings):
                if "::" not in chunk:
                    raise ValueError(f"Chunk {chunk} does not have a field_name::content format")
                field_name, content = chunk.split("::", 2)
                if field_name not in marqo_document[index_constants.MARQO_DOC_TENSORS]:
                    marqo_document[index_constants.MARQO_DOC_TENSORS][field_name] = dict()
                marqo_document[index_constants.MARQO_DOC_TENSORS][field_name][content] = embedding

        if return_highlights and self.match_features:
            marqo_document[index_constants.MARQO_DOC_HIGHLIGHTS] = self._extract_highlights()

        return marqo_document

    def _extract_highlights(self):
        if not self.match_features:
            raise ValueError("No match features found in the document")
        try:
            chunk_index: int = int(list(self.match_features[f"closest({unstructured_common.VESPA_DOC_EMBEDDINGS})"]\
                                   ["cells"].keys())[0])
        except KeyError:
            raise ValueError("No match features found in the document")
        field_name, content = self.marqo_chunks[chunk_index].split("::", 2)
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
        instance = cls(id=str(copied[index_constants.MARQO_DOC_ID]))
        del copied[index_constants.MARQO_DOC_ID]

        for key, value in copied.items():
            if key in [index_constants.MARQO_DOC_EMBEDDINGS, index_constants.MARQO_DOC_CHUNKS]:
                continue
            if isinstance(value, str):
                if len(value) <= unstructured_constants.SHORT_STRING_THRESHOLD:
                    instance.fields.short_string_fields[key] = value
                else:
                    instance.fields.long_string_fields[key] = value
                instance.fields.strings.append(value)
            elif isinstance(value, list) and all(isinstance(elem, str) for elem in value):
                instance.fields.string_arrays.extend([f"{key}::{element}" for element in value])
            elif isinstance(value, int):
                instance.fields.int_fields[key] = value
                instance.fields.score_modifiers_fields[key] = value
            elif isinstance(value, float):
                instance.fields.float_fields[key] = value
                instance.fields.score_modifiers_fields[key] = value

        instance.fields.marqo__id = instance.id
        instance.fields.vespa_embeddings = copied.get(index_constants.MARQO_DOC_EMBEDDINGS, {})
        instance.fields.vespa_chunks = copied.get(index_constants.MARQO_DOC_CHUNKS, [])
        return instance

    def to_vespa_document(self) -> Dict[str, Any]:
        """Convert VespaDocumentObject to a Vespa document.
        Empty fields are removed from the document."""
        return {index_constants.VESPA_DOC_ID: self.id,
                index_constants.VESPA_DOC_FIELDS: self.fields.to_vespa_dictionary()}

    def to_marqo_document(self, return_highlights: bool = False) -> Dict[str, Any]:
        """Convert VespaDocumentObject back to the original document structure."""
        return self.fields.to_marqo_doc(return_highlights=return_highlights)



