import json
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union
import uuid

import numpy as np
from pydantic.dataclasses import dataclass
from pydantic import Field, validator, BaseModel, root_validator

from marqo.tensor_search import constants, enums, validation, utils
import marqo.errors as errors
from marqo.tensor_search.models.private_models import ModelAuth


class Weights(BaseModel):
    __root__: Dict[str, float]  # Allowing only string keys and number values

    @root_validator(pre=True, allow_reuse=True)
    def check_keys(cls, values):
        if not all(isinstance(i, str) for i in values.keys()):
            raise ValueError('All keys must be strings')
        if not all(isinstance(i, (int, float)) for i in values.values()):
            raise ValueError('All values must be numbers')
        return values


class MappingObject(BaseModel):
    type: enums.MappingsObjectType
    weights: Weights = None  # Default value

    class Config:
        extra = 'allow'

    @validator('weights', always=True, allow_reuse=True)
    def check_weights(cls, weights, values, **kwargs):
        """Ensure that weights is provided if and only if the type is 'MappingsObjectType.multimodal_combination'."""
        is_multimodal = values['type'] == enums.MappingsObjectType.multimodal_combination
        if is_multimodal and weights is not None:
            raise errors.InvalidArgError(
                f"Error validating multimodal combination object. Reason: 'weights' is not provided"
                f"\nRead about the mappings object here: https://docs.marqo.ai/0.0.15/API-Reference/mappings/"
            )
        elif not is_multimodal and weights is None:
            raise errors.InvalidArgError(
                f"Error validating mappings. Mapping is not multimodal combination object. Reason: field is not multimodal, but weights provided."
                f"\nRead about the mappings object here: https://docs.marqo.ai/0.0.15/API-Reference/mappings/"
            )
        return weights

class Document(BaseModel):
    _id: Optional[str] = None

    class Config:
        extra = 'allow'

    @validator('*', pre=True, allow_reuse=True)
    def validate_field_names(self, v, values, field):
        if field.name == "_id":
            return

        validation.validate_field_name(field)

    @root_validator(allow_reuse=True)
    def validate_id(cls, values):
        if values.get("_id") is not None:
            validation.validate_id(values.get("_id"))

        return values

    @root_validator(allow_reuse=True)
    def validate_document_size(cls, values):
        """Ensure document is less than MARQO_MAX_DOC_BYTES bytes (when set)."""

        max_doc_size = utils.read_env_vars_and_defaults(var=enums.EnvVars.MARQO_MAX_DOC_BYTES)
        if max_doc_size is None:    
            return

        try:
            serialized = json.dumps(values)
        except TypeError as e:
            raise errors.InvalidArgError(
                f"Unable to index document: it is not serializable! Document: `{values}` "
            )

        if len(serialized) > int(max_doc_size):
            maybe_id = f" _id:`{values['_id']}`" if '_id' in values else ''
            raise errors.DocTooLargeError(
                f"Document{maybe_id} with length `{len(serialized)}` exceeds "
                f"the allowed document size limit of [{max_doc_size}]."
            )
    
        return values

    def get_or_generate_id(self, id_gen: Callable[[None], str] = lambda: str(uuid.uuid4())) -> str:
        """Gets the '_id' or generates AND sets to self, a UUID."""
        if self._id is not None:
            return self._id
        
        self._id = id_gen()
        validation.validate_id(self._id)
        return self._id
    
    def get_chunk_values_for_filtering(self) -> Dict[str, Any]:
        """
        Metadata can be calculated here at the doc level. Only add chunk 
        values which are string, boolean, numeric or dictionary. Dictionary
        keys will be store in a list.
        """
        return dict([
            (k, v)for k, v in self.__dict__.items() if k != "_id" and isinstance(v, (str, float, bool, int, list, dict))
        ])


class AddDocsParams(BaseModel):
    """Represents the parameters of the tensor_search.add_documents() function

    Params:
        index_name: name of the index
        docs: List of documents
        auto_refresh: Set to False if indexing lots of docs
        non_tensor_fields: List of fields, within documents to not create tensors for. Default to
          make tensors for all fields.
        use_existing_tensors: Whether to use the vectors already in doc (for update docs)
        device: Device used to carry out the document update.
        update_mode: {'replace' | 'update'}. If set to replace (default) just
        image_download_thread_count: number of threads used to concurrently download images
        image_download_headers: headers to authenticate image download
        mappings: a dictionary used to handle all the object field content in the doc,
            e.g., multimodal_combination field
        model_auth: an object used to authorise downloading an object from a datastore

    """

    index_name: str
    auto_refresh: bool

    # this should only accept Sequences of dicts, but currently validation lies elsewhere
    docs: List[Document]= Field(default_factory=list)
    
    non_tensor_fields: List = Field(default_factory=list)
    device: Optional[str] = None
    update_mode: Optional[str] = "replace"
    image_download_thread_count: int = 20
    image_download_headers: dict = Field(default_factory=dict)
    use_existing_tensors: bool = False
    mappings: Optional[Dict[str, MappingObject]] = None
    model_auth: Optional[ModelAuth] = None

    class Config:
        arbitrary_types_allowed = True

    @validator('docs', always=True)
    def docs_isnt_empty(cls, v, values, field):
        if len(v) == 0:
            raise errors.BadRequestError(message="Received empty add documents request")

    @validator('use_existing_tensors', allow_reuse=True)
    def use_existing_tensors_cannot_be_in_update(cls, y, values, **kwargs):
        """Ensure if use_existing_tensors=True, then update_mode is replace (not update)"""
        if y and values['update_mode'] != "replace":
            raise errors.InvalidArgError("use_existing_tensors=True is only available for add and replace documents,"
                                     "not for add and update!")
    
    @validator('update_mode', pre=True, always=True, allow_reuse=True)
    def validate_update_mode_is_valid(cls, v, values, field):
        valid_update_modes = ('update', 'replace')
        if v not in valid_update_modes:
            raise errors.InvalidArgError(message=f"Unknown update_mode `{v}` received! Valid update modes: {valid_update_modes}")

    def batch_size(self) -> int:
        return len(self.docs)
    
    def indexing_instructions(self) -> Dict[str, Any]:
        key = "index" if self.update_mode == 'replace' else "update"
        return {
            key: {"_index": self.index_name,}
        }

    def get_doc_ids_uniquely(self) -> List[str]:
        """Get unique, non-None, document ids."""
        doc_ids = []
        seen_ids = set()
        for doc in reversed(self.docs):
            doc_id = doc.get("_id")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc["_id"])
                doc_ids.append(doc["_id"])

        # As we appended the docs in reversed order, reverse it again to maintain the original order
        return doc_ids[::-1]