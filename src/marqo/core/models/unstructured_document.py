from typing import List, Dict, Any
from copy import deepcopy

from marqo.core.models.strict_base_model import StrictBaseModel
from pydantic import Field


SHORT_STRING_THRESHOLD = 20


class UnstructuredVespaDocumentFields(StrictBaseModel):
    """A class with fields that are common to all Vespa documents."""
    strings: List[str] = Field(default_factory=list)
    long_string_fields: Dict[str, str] = Field(default_factory=dict)
    short_string_fields: Dict[str, str] = Field(default_factory=dict)
    string_arrays: List[str] = Field(default_factory=list)
    int_fields: Dict[str, int] = Field(default_factory=dict)
    float_fields: Dict[str, float] = Field(default_factory=dict)
    marqo_chunks: List[str] = Field(default_factory=list)
    marqo_embeddings: Dict = Field(default_factory=dict)

    def to_vespa_dictionary(self) -> Dict[str, Any]:
        return self.dict(exclude_none=True)

    def to_marqo_dictionary(self) -> Dict[str, Any]:
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
        return marqo_document


class UnstructuredIndexDocument(StrictBaseModel):
    """A helper class to handle the conversion between Vespa and Marqo documents for an unstructured index.

    The object can be instantiated from a Marqo document using the from_marqo_document method,
    or can be instantiated from a Vespa document using the from_vespa_document method.
    """
    id: str
    fields: UnstructuredVespaDocumentFields = Field(default_factory=UnstructuredVespaDocumentFields())

    def __init__(self, *args, **kwargs):
        raise RuntimeError("UnstructuredIndexDocument is not meant to be instantiated directly. Use "
                           "from marqo_document or from_vespa_document instead.")

    @classmethod
    def from_vespa_document(cls, document: Dict) -> "UnstructuredIndexDocument":
        """Instantiate an UnstructuredIndexDocument from a Vespa document."""
        return cls(id=[document["id"]], fields=UnstructuredVespaDocumentFields(**document.get("fields", {})))

    @classmethod
    def from_marqo_document(cls, document: Dict) -> "UnstructuredIndexDocument":
        """Instantiate an UnstructuredIndexDocument from a Marqo document."""
        # TODO Validation for Marqo document
        copied = deepcopy(document)

        instance = cls(id=str(copied["_id"]))
        del copied["_id"]

        for key, value in copied.items():
            if key in ["marqo_embeddings", "marqo_chunks"]:
                continue
            if isinstance(value, str):
                if len(value) <= SHORT_STRING_THRESHOLD:
                    instance.fields.short_string_fields[key] = value
                else:
                    instance.fields.long_string_fields[key] = value
                instance.fields.strings.append(value)
            elif isinstance(value, list) and all(isinstance(elem, str) for elem in value):
                instance.fields.string_arrays.extend([f"{key}::{element}" for element in value])
            elif isinstance(value, int):
                instance.fields.int_fields[key] = value
            elif isinstance(value, float):
                instance.fields.float_fields[key] = value

        instance.fields.marqo_embeddings = copied.get("marqo_embeddings", {})
        instance.fields.marqo_chunks = copied.get("marqo_chunks", [])
        return instance

    def to_vespa_document(self) -> Dict[str, Any]:
        """Convert VespaDocumentObject to a Vespa document.
        Empty fields are removed from the document."""
        return {"id": self.id, "fields": self.fields.to_vespa_dictionary()}

    def to_marqo_document(self) -> Dict[str, Any]:
        """Convert VespaDocumentObject back to the original document structure."""
        return {"_id": self.id, **self.fields.to_marqo_dictionary()}

