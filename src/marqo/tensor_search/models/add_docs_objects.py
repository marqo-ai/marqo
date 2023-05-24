import json
from typing import Any, Callable, Dict, List, Optional
import uuid

from pydantic import Field, validator, BaseModel, root_validator, ValidationError

from marqo.tensor_search import enums, validation, utils
import marqo.errors as errors
from marqo.tensor_search.models.private_models import ModelAuth


class Weights(BaseModel):
    class Config:
        extra = 'allow'

    @root_validator(pre=True, allow_reuse=True)
    def check_keys(cls, values):
        """Ensure all keys map to floats"""
        if not all(isinstance(i, str) for i in values.keys()):
            raise ValueError('All keys must be strings')
        if not all(isinstance(i, (int, float)) and not isinstance(i, bool) for i in values.values()):
            # Note: `isinstance(True, (int, float)) == True`
            raise ValueError('All values must be numbers')
        return values


class MappingObject(BaseModel):
    type: enums.MappingsObjectType
    weights: Weights = None  # Default value

    def __init__(self, **data):
        try:
            super().__init__(**data)
        except ValidationError as e:
            raise errors.InvalidArgError(message=e.json())

    class Config:
        extra = 'forbid'

    @validator('weights', always=True, allow_reuse=True)
    def check_weights(cls, weights, values, **kwargs):
        """Ensure that weights is provided if and only if the type is 'MappingsObjectType.multimodal_combination'."""
        is_multimodal = values.get('type', None) == enums.MappingsObjectType.multimodal_combination
        if is_multimodal and weights is None:
            raise errors.InvalidArgError(
                f"Error validating multimodal combination object. Reason: 'weights' is not provided"
                f"\nRead about the mappings object here: https://docs.marqo.ai/0.0.15/API-Reference/mappings/"
            )
        elif not is_multimodal and weights is not None:
            raise errors.InvalidArgError(
                f"Error validating mappings. Mapping is not multimodal combination object. Reason: field is not multimodal, but weights provided."
                f"\nRead about the mappings object here: https://docs.marqo.ai/0.0.15/API-Reference/mappings/"
            )
        return weights

class Document(BaseModel):
    _id: Optional[str] = None

    class Config:
        extra = 'allow'

    # @validator('*', pre=True, allow_reuse=True)
    def validate_field_names(self, v, values, field):
        if field.name == "_id":
            return

        validation.validate_field_name(field)

    # @root_validator(allow_reuse=True)
    def validate_id(cls, values):
        if "_id" in values.keys():
            validation.validate_id(values.get("_id"))

        return values

    # @root_validator(allow_reuse=True)
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

    def _validate_field_name(self, key: str):
        if key == "_id":
            return 
        
        validation.validate_field_name(key)

    def validate_document(self) -> None:
        """
        Validate the document object.
        """
        self.validate_document_size(self.dict())
        self.validate_id(self.dict())
        [self._validate_field_name(k) for (k, _) in self.dict().items()]

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
        return v

    @validator('use_existing_tensors', allow_reuse=True)
    def use_existing_tensors_cannot_be_in_update(cls, y, values, **kwargs):
        """Ensure if use_existing_tensors=True, then update_mode is replace (not update)"""
        if y and values['update_mode'] != "replace":
            raise errors.InvalidArgError("use_existing_tensors=True is only available for add and replace documents,"
                                     "not for add and update!")
        return y 
    
    @validator('update_mode', pre=True, always=True, allow_reuse=True)
    def validate_update_mode_is_valid(cls, v, values, field):
        valid_update_modes = ('update', 'replace')
        if v not in valid_update_modes:
            raise errors.InvalidArgError(message=f"Unknown update_mode `{v}` received! Valid update modes: {valid_update_modes}")
        return v

    def batch_size(self) -> int:
        return len(self.docs)
    
    def indexing_instructions(self, doc_id: str) -> Dict[str, Any]:
        indexing_instructions = {
            "index" if self.update_mode == 'replace' else "update": {"_index": self.index_name,}
        }
        if self.update_mode == "replace":
            indexing_instructions["index"]["_id"] = doc_id
        else:
            indexing_instructions["update"]["_id"] = doc_id
        return indexing_instructions
    
    def get_doc_ids_uniquely(self) -> List[str]:
        """Get unique, non-None, document ids."""
        doc_ids = []
        seen_ids = [] # Cannot use set in case doc._id is not a valid ID and is not hashable
        for doc in reversed(self.docs):
            doc_id = doc._id
            if doc_id and doc_id not in seen_ids:
                seen_ids.append(doc_id)
                doc_ids.append(doc_id)

        # As we appended the docs in reversed order, reverse it again to maintain the original order
        return doc_ids[::-1]
    
    def create_anew(self, **new_kwargs: Dict[str, Any]) -> "AddDocsParams":
        new_dict = {**self.dict(), **new_kwargs}
        return AddDocsParams(**new_dict)