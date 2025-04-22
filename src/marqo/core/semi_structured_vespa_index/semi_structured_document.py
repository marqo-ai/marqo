import json
import uuid
from typing import List, Dict, Any, Union, Optional

from pydantic import Field

from marqo.base_model import MarqoBaseModel
from marqo.core import constants as index_constants, constants
from marqo.core.exceptions import VespaDocumentParsingError, MarqoDocumentParsingError, InvalidFieldNameError, \
    InvalidTensorFieldError
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex
from marqo.core.semi_structured_vespa_index import common
from marqo.core.semi_structured_vespa_index.common import VESPA_DOC_FIELD_TYPES, STRING_ARRAY
from marqo.core.semi_structured_vespa_index.marqo_field_types import MarqoFieldTypes
from marqo.core.unstructured_vespa_index.common import MARQO_DOC_MULTIMODAL_PARAMS, MARQO_DOC_MULTIMODAL_PARAMS_WEIGHTS


def generate_uuid_str() -> str:
    return str(uuid.uuid4()).replace('-', '')


class SemiStructuredVespaDocumentFields(MarqoBaseModel):
    """A class with fields that are common to all Vespa documents."""
    marqo__id: str = Field(alias=common.VESPA_FIELD_ID)
    version_uuid: Optional[str] = Field(default_factory=None, alias=common.VESPA_DOC_VERSION_UUID)

    short_string_fields: Dict[str, str] = Field(default_factory=dict, alias=common.SHORT_STRINGS_FIELDS)
    string_arrays: List[str] = Field(default_factory=list, alias=common.STRING_ARRAY) # Indexes created pre marqo version 2.16 will have string arrays stored as a list of strings
    int_fields: Dict[str, int] = Field(default_factory=dict, alias=common.INT_FIELDS)
    bool_fields: Dict[str, int] = Field(default_factory=dict, alias=common.BOOL_FIELDS)
    float_fields: Dict[str, float] = Field(default_factory=dict, alias=common.FLOAT_FIELDS)
    score_modifiers_fields: Dict[str, Any] = Field(default_factory=dict, alias=common.SCORE_MODIFIERS)
    vespa_multimodal_params: Dict[str, str] = Field(default_factory=dict, alias=common.VESPA_DOC_MULTIMODAL_PARAMS)
    field_types: Dict[str, str] = Field(default_factory=dict, alias=common.VESPA_DOC_FIELD_TYPES)


class SemiStructuredVespaDocument(MarqoBaseModel):
    """A helper class to handle the conversion between Vespa and Marqo documents for a semi-structured index.
    The object can be instantiated from a Marqo document using the from_marqo_document method,
    or can be instantiated from a Vespa document using the from_vespa_document method.
    """
    id: str
    fixed_fields: SemiStructuredVespaDocumentFields = Field(default_factory=SemiStructuredVespaDocumentFields)
    text_fields: dict = Field(default_factory=dict)
    tensor_fields: dict = Field(default_factory=dict)
    vector_counts: int = Field(default=0, alias=common.FIELD_VECTOR_COUNT)
    match_features: Dict[str, Any] = Field(default_factory=dict, alias=common.VESPA_DOC_MATCH_FEATURES)
    string_array_fields: Dict[str, List[str]] = Field(default_factory=dict)
    index_supports_partial_updates: bool = False

    # For hybrid search
    raw_tensor_score: float = None
    raw_lexical_score: float = None

    _VESPA_DOC_FIELDS = "fields"
    _VESPA_DOC_ID = "id"

    @classmethod
    def from_vespa_document(cls, document: Dict, marqo_index: SemiStructuredMarqoIndex) -> "SemiStructuredVespaDocument":
        """
        Instantiate an SemiStructuredVespaDocument from a Vespa document.
        Used in get_document_by_id or get_documents_by_ids
        """
        fields = document.get(cls._VESPA_DOC_FIELDS, {})
        tensor_fields = {}
        text_fields = {}
        string_arrays_dict = {}
        string_arrays_list = []

        def process_field(field_name: str, fields: Dict) -> None:
            """Helper function to process individual tensor fields, lexical fields and populate the appropriate dictionaries"""
            if field_name in marqo_index.tensor_subfield_map:
                tensor_fields[field_name] = fields[field_name]
            elif field_name in marqo_index.lexical_field_map:
                # Lexical fields are returned with prefixed name from get_by_ids
                text_field_name = marqo_index.lexical_field_map[field_name].name
                text_fields[text_field_name] = fields[field_name]
            elif field_name in marqo_index.field_map:
                # Lexical fields are returned with original name from search
                text_fields[field_name] = fields[field_name]

        if marqo_index.index_supports_partial_updates:
            for field_name in fields:
                # Process tensor and text fields
                process_field(field_name, fields)

                # Handle string arrays separately
                # In case of indexes which  support partial updates (i.e indexes created with Marqo version post 2.16.0), string arrays are stored in Vespa like
                # 'marqo__string_array_field_name_1': ['element1', 'element2', ...]
                # 'marqo__string_array_field_name_2': ['element3', 'element4', ...]
                # Here we will collect all such string array fields and put them in string_arrays_dict, which will later be used  to construct the SemiStructuredVespaDocument object.
                if field_name.startswith(common.STRING_ARRAY+'_'):
                    string_array_field_key = field_name.replace(common.STRING_ARRAY+'_', '')
                    string_array_field_value = fields[field_name]
                    string_arrays_dict[string_array_field_key] = string_array_field_value

            fixed_fields = SemiStructuredVespaDocumentFields.construct(
                marqo__id=cls.extract_field(fields, common.VESPA_FIELD_ID, None),
                version_uuid=cls.extract_field(fields, common.VESPA_DOC_VERSION_UUID, None),
                short_string_fields=cls.extract_field(fields, common.SHORT_STRINGS_FIELDS, dict()),
                string_arrays=cls.extract_field(fields, common.STRING_ARRAY, list()),
                int_fields=cls.extract_field(fields, common.INT_FIELDS, dict()),
                bool_fields=cls.extract_field(fields, common.BOOL_FIELDS, dict()),
                float_fields=cls.extract_field(fields, common.FLOAT_FIELDS, dict()),
                score_modifiers_fields=cls.extract_field(fields, common.SCORE_MODIFIERS, dict()),
                vespa_multimodal_params=cls.extract_field(fields, common.VESPA_DOC_MULTIMODAL_PARAMS, dict()),
                field_types=cls.extract_field(fields, VESPA_DOC_FIELD_TYPES, dict())
            )

            return cls(id=document[cls._VESPA_DOC_ID],
                    fixed_fields=fixed_fields,
                    tensor_fields=tensor_fields,
                    text_fields=text_fields,
                    string_array_fields = string_arrays_dict,
                    raw_tensor_score=cls.extract_field(fields, common.VESPA_DOC_HYBRID_RAW_TENSOR_SCORE, None),
                    raw_lexical_score=cls.extract_field(fields, common.VESPA_DOC_HYBRID_RAW_LEXICAL_SCORE, None),
                    match_features=cls.extract_field(fields, common.VESPA_DOC_MATCH_FEATURES, dict()),
                    vector_counts=cls.extract_field(fields, common.FIELD_VECTOR_COUNT, 0),
                    index_supports_partial_updates=True)
        else:
            # For older versions, just process tensor and text fields
            for field_name in fields:
                process_field(field_name, fields)

                # Handle string arrays separately
                # In case of indexes which don't support partial updates (i.e indexes created with Marqo version pre 2.16.0), string arrays are stored like
                # 'marqo__string_array': ['field_name_1::element1', 'field_name_1::element2', 'field_name_2::element3', 'field_name_2::element4', ...], inside the Vespa document.
                # Here we will just take the list defined under marqo__string_array and put under string_arrays_list, which will later be used  to construct the SemiStructuredVespaDocumentFields object.
                # The SemiStructuredVespaDocumentFields object will later be used to construct SemiStructuredVespaDocument object.
                # Later when converting SemiStructuredVespaDocument object to Marqo document (which is a dictionary of fields to be returned to the user), we will process this list & convert it
                # to a dictionary by splitting the list element by '::', making the prefix before '::' as the key and suffix after '::' as the value.
                if field_name == common.STRING_ARRAY:
                    string_arrays_list = fields[field_name]
            fixed_fields = SemiStructuredVespaDocumentFields.construct(
                marqo__id=cls.extract_field(fields, common.VESPA_FIELD_ID, None),
                short_string_fields=cls.extract_field(fields, common.SHORT_STRINGS_FIELDS, dict()),
                string_arrays=string_arrays_list,
                int_fields=cls.extract_field(fields, common.INT_FIELDS, dict()),
                bool_fields=cls.extract_field(fields, common.BOOL_FIELDS, dict()),
                float_fields=cls.extract_field(fields, common.FLOAT_FIELDS, dict()),
                score_modifiers_fields=cls.extract_field(fields, common.SCORE_MODIFIERS, dict()),
                vespa_multimodal_params=cls.extract_field(fields, common.VESPA_DOC_MULTIMODAL_PARAMS, dict()),
                field_types=cls.extract_field(fields, VESPA_DOC_FIELD_TYPES, dict())
            )

            return cls(id=document[cls._VESPA_DOC_ID],
                    fixed_fields=fixed_fields,
                    tensor_fields=tensor_fields,
                    text_fields=text_fields,
                    raw_tensor_score=cls.extract_field(fields, common.VESPA_DOC_HYBRID_RAW_TENSOR_SCORE, None),
                    raw_lexical_score=cls.extract_field(fields, common.VESPA_DOC_HYBRID_RAW_LEXICAL_SCORE, None),
                    match_features=cls.extract_field(fields, common.VESPA_DOC_MATCH_FEATURES, dict()),
                    vector_counts=cls.extract_field(fields, common.FIELD_VECTOR_COUNT, 0),
                    index_supports_partial_updates=False)

    @classmethod
    def extract_field(cls, fields, name: str, default: Any):
        return fields[name] if name in fields else default

    @classmethod
    def from_marqo_document(cls, document: Dict, marqo_index: SemiStructuredMarqoIndex) -> "SemiStructuredVespaDocument":
        """
        Creates a SemiStructuredVespaDocument object from a Marqo document.

        Args:
            document (Dict): A dictionary representing a valid Marqo document. Must contain a '_id' field
                and can include various field types like strings, booleans, numbers, arrays and tensors.
            marqo_index (SemiStructuredMarqoIndex): The Marqo index object that this document belongs to.
                Used to determine index capabilities and settings.

        Returns:
            SemiStructuredVespaDocument: A new instance containing the document data structured for Vespa.

        Raises:
            MarqoDocumentParsingError: If the document is missing required fields or contains invalid data.

        Example:
            doc = {
                "_id": "doc1",
                "title": "Sample Document",
                "tags": ["tag1", "tag2"],
                "rating": 4.5
            }
            vespa_doc = SemiStructuredVespaDocument.from_marqo_document(doc, index)
        """
        index_supports_partial_updates = marqo_index.index_supports_partial_updates
        if index_constants.MARQO_DOC_ID not in document:
            # Please note we still use unstructured in the error message since it will be exposed to user
            raise MarqoDocumentParsingError(
                f"Unstructured Marqo document does not have a {index_constants.MARQO_DOC_ID} field. "
                f"This should be assigned for a valid document")

        doc_id = document[index_constants.MARQO_DOC_ID]
        instance = cls(
            id=doc_id,
            fixed_fields=SemiStructuredVespaDocumentFields(
                marqo__id=doc_id,
                version_uuid=generate_uuid_str() if index_supports_partial_updates else None,
            ),
            index_supports_partial_updates=index_supports_partial_updates
        )

        # Process regular fields
        cls._process_regular_fields(document, instance, marqo_index, doc_id)

        # Process tensor fields if present
        vector_count = cls._process_tensor_fields(document, instance, marqo_index)
        instance.vector_counts = vector_count

        # Add multimodal params if present
        instance.fixed_fields.vespa_multimodal_params = document.get(common.MARQO_DOC_MULTIMODAL_PARAMS, {})

        return instance

    @classmethod
    def _process_regular_fields(cls, document: dict, instance, marqo_index: SemiStructuredMarqoIndex, doc_id: str):
        """Process non-tensor fields in the document"""
        for field_name, field_content in document.items():
            if field_name in [index_constants.MARQO_DOC_ID, constants.MARQO_DOC_TENSORS]:
                continue
            try:
                cls._handle_field_content(field_name, field_content, instance, marqo_index)
            except Exception as e:
                raise MarqoDocumentParsingError(
                    f"Error processing field '{field_name}' in document {doc_id}: {str(e)}")

    @classmethod
    def _handle_field_content(cls, field_name: str, field_content: Union[str, bool, list, int, float, dict], instance,
                              marqo_index: SemiStructuredMarqoIndex):
        """Handle different field content types"""
        if isinstance(field_content, str):
            cls._handle_string_field(field_name, field_content, instance, marqo_index)
        elif isinstance(field_content, bool):
            cls._handle_bool_field(field_name, field_content, instance)
        elif isinstance(field_content, list) and all(isinstance(elem, str) for elem in field_content):
            cls._handle_string_array_field(field_name, field_content, instance)
        elif isinstance(field_content, (int, float)):
            cls._handle_numeric_field(field_name, field_content, instance)
        elif isinstance(field_content, dict):
            cls._handle_dict_field(field_name, field_content, instance)
        else:
            raise MarqoDocumentParsingError(f"Unsupported type {type(field_content)}")

    @classmethod
    def _handle_string_field(cls, field_name: str, field_content: str, instance, marqo_index: SemiStructuredMarqoIndex):
        if field_name not in marqo_index.field_map:
            raise MarqoDocumentParsingError(f'Field {field_name} is not in index {marqo_index.name}')
        
        field = marqo_index.field_map[field_name]
        instance.text_fields[field.lexical_field_name] = field_content
        
        if len(field_content) <= marqo_index.filter_string_max_length:
            instance.fixed_fields.short_string_fields[field_name] = field_content
            
        if instance.index_supports_partial_updates:
            instance.fixed_fields.field_types[field_name] = MarqoFieldTypes.STRING.value

    @classmethod
    def _handle_bool_field(cls, field_name: str, field_content: bool, instance):
        instance.fixed_fields.bool_fields[field_name] = int(field_content)
        if instance.index_supports_partial_updates:
            instance.fixed_fields.field_types[field_name] = MarqoFieldTypes.BOOL.value

    @classmethod
    def _handle_string_array_field(cls, field_name: str, field_content: List[str], instance):
        if instance.index_supports_partial_updates:
            instance.string_array_fields[field_name] = field_content
            instance.fixed_fields.field_types[field_name] = MarqoFieldTypes.STRING_ARRAY.value
        else: 
            instance.fixed_fields.string_arrays.extend([f"{field_name}::{element}" for element in field_content])

    @classmethod
    def _handle_numeric_field(cls, field_name: str, field_content: Union[int, float], instance):
        if isinstance(field_content, int):
            instance.fixed_fields.int_fields[field_name] = field_content
            if instance.index_supports_partial_updates:
                instance.fixed_fields.field_types[field_name] = MarqoFieldTypes.INT.value
        else:  # float
            instance.fixed_fields.float_fields[field_name] = field_content
            if instance.index_supports_partial_updates:
                instance.fixed_fields.field_types[field_name] = MarqoFieldTypes.FLOAT.value
        instance.fixed_fields.score_modifiers_fields[field_name] = field_content

    @classmethod
    def _handle_dict_field(cls, field_name: str, field_content: Dict[str, Union[int, float]], instance):
        for k, v in field_content.items():
            field_key = f"{field_name}.{k}" # field_key is the flattened field name for a dict field.
            if isinstance(v, int):
                instance.fixed_fields.int_fields[field_key] = v
                instance.fixed_fields.score_modifiers_fields[field_key] = v
                if instance.index_supports_partial_updates:
                    instance.fixed_fields.field_types[field_key] = MarqoFieldTypes.INT_MAP.value
                    instance.fixed_fields.field_types[field_name] = MarqoFieldTypes.INT_MAP.value # Marking the overall dict field as a int_map_entry field as well
            elif isinstance(v, float):
                instance.fixed_fields.float_fields[field_key] = float(v)
                instance.fixed_fields.score_modifiers_fields[field_key] = v
                if instance.index_supports_partial_updates:
                    instance.fixed_fields.field_types[field_key] = MarqoFieldTypes.FLOAT_MAP.value
                    instance.fixed_fields.field_types[field_name] = MarqoFieldTypes.FLOAT_MAP.value # Marking the overall dict field as a float_map_entry field as well

    @classmethod
    def _process_tensor_fields(cls, document: Dict, instance, marqo_index: SemiStructuredMarqoIndex):
        """Process tensor fields in the document"""
        vector_count = 0
        if constants.MARQO_DOC_TENSORS in document:
            for marqo_tensor_field, tensor_value in document[constants.MARQO_DOC_TENSORS].items():
                if instance.index_supports_partial_updates:
                    instance.fixed_fields.field_types[marqo_tensor_field] = MarqoFieldTypes.TENSOR.value # Set field_types as tensor for tensor fields

                    multimodal_params = document.get(MARQO_DOC_MULTIMODAL_PARAMS)
                    if multimodal_params is not None and multimodal_params.get(marqo_tensor_field) is not None: # Set field_types as tensor for sub-fields of multimodal combo fields
                        try:
                            multimodal_params = json.loads(document.get(MARQO_DOC_MULTIMODAL_PARAMS).get(marqo_tensor_field))
                            multimodal_combo_sub_fields = multimodal_params.get(MARQO_DOC_MULTIMODAL_PARAMS_WEIGHTS).keys()
                            for sub_field in multimodal_combo_sub_fields:
                                instance.fixed_fields.field_types[sub_field] = MarqoFieldTypes.TENSOR.value
                        except json.JSONDecodeError as e:
                            raise MarqoDocumentParsingError(f"Error parsing multimodal params for field {marqo_tensor_field}: {str(e)}")

                cls._verify_marqo_tensor_field_name(marqo_tensor_field, marqo_index)
                cls._verify_marqo_tensor_field(marqo_tensor_field, tensor_value)

                # If chunking an image, chunks will be a list of tuples, hence the str(c)
                chunks = [str(c) for c in tensor_value[constants.MARQO_DOC_CHUNKS]]
                embeddings = tensor_value[constants.MARQO_DOC_EMBEDDINGS]
                vector_count += len(embeddings)

                index_tensor_field = marqo_index.tensor_field_map[marqo_tensor_field]
                instance.tensor_fields[index_tensor_field.chunk_field_name] = chunks
                instance.tensor_fields[index_tensor_field.embeddings_field_name] = \
                    {f'{i}': embeddings[i] for i in range(len(embeddings))}

        return vector_count

    def to_vespa_document(self) -> Dict[str, Any]:
        """
        Converts this SemiStructuredVespaDocument object to a Vespa document format.
        
        @return: Dictionary containing the Vespa document representation with document ID and fields
        
        The returned document will have empty fields removed. The document structure follows Vespa's 
        expected format with a document ID and fields dictionary containing:
        - Fixed fields (integers, floats, booleans etc)
        - Text fields
        - Tensor fields 
        - Vector count
        - String arrays (handled differently based on partial update support)
        """
        vespa_fields = {
            **{k: v for k, v in self.fixed_fields.dict(exclude_none=True, by_alias=True).items() if v or v == 0},
            **self.text_fields,
            **self.tensor_fields,
            common.FIELD_VECTOR_COUNT: self.vector_counts,
        }

        if self.index_supports_partial_updates:
            if self.string_array_fields is not None:
                for string_array_key, string_array_value in self.string_array_fields.items():
                    key = f'{STRING_ARRAY}_{string_array_key}'
                    vespa_fields[key] = string_array_value
        else:
            vespa_fields[common.STRING_ARRAY] = self.fixed_fields.string_arrays

        return {self._VESPA_DOC_ID: self.id, self._VESPA_DOC_FIELDS: vespa_fields}

    def to_marqo_document(self, marqo_index: SemiStructuredMarqoIndex) -> Dict[str, Any]:
        """
        Convert SemiStructuredVespaDocument object to marqo document structure.

        Args:
            marqo_index: The SemiStructuredMarqoIndex instance containing index configuration

        Returns:
            Dict[str, Any]: A dictionary representing the marqo document format containing:
                - String array fields (handled differently for pre/post 2.16 indexes)
                - Integer and float fields
                - Boolean fields 
                - Document ID
                - Text fields
                - Tensor fields with chunks and embeddings
        """
        marqo_document = {}

        if self.index_supports_partial_updates and self.string_array_fields:
            # self.string_array_fields is a dictionary, Post 2.16 indexes will have string arrays stored as a map of string to list of strings.
            marqo_document.update(self.string_array_fields)
        else: # Pre 2.16 indexes will have string arrays stored as a list of strings in the SemiStructuredVespaDocumentsFields object under a field called "string_arrays".
            for string_array in self.fixed_fields.string_arrays:
                string_array_key, string_array_value = string_array.split("::", 1) # String_array_key will be string in this case, and string_array_value will be a single string in this case.
                if string_array_key not in marqo_document:
                    marqo_document[string_array_key] = []
                marqo_document[string_array_key].append(string_array_value)

        # Add int and float fields back
        # Please note that int-map and float-map fields are flattened in the result. The correct behaviour is to convert
        # them back to the format when they are indexed. We will keep the behaviour as is to avoid breaking changes.
        marqo_document.update(self.fixed_fields.int_fields)
        marqo_document.update(self.fixed_fields.float_fields)

        marqo_document.update({k: bool(v) for k, v in self.fixed_fields.bool_fields.items()})
        marqo_document[index_constants.MARQO_DOC_ID] = self.fixed_fields.marqo__id
        # Note: We are not adding field_types & version_uuid to the document because
        # it's a field for internal Marqo use only.

        # text fields
        for field_name, field_content in self.text_fields.items():
            marqo_document[field_name] = field_content

        # tensor fields
        for field_name, field_content in self.tensor_fields.items():
            tensor_field = marqo_index.tensor_subfield_map[field_name]

            if constants.MARQO_DOC_TENSORS not in marqo_document:
                marqo_document[constants.MARQO_DOC_TENSORS] = dict()
            if tensor_field.name not in marqo_document[constants.MARQO_DOC_TENSORS]:
                marqo_document[constants.MARQO_DOC_TENSORS][tensor_field.name] = dict()

            if field_name == tensor_field.chunk_field_name:
                marqo_document[constants.MARQO_DOC_TENSORS][tensor_field.name][constants.MARQO_DOC_CHUNKS] = field_content
            elif field_name == tensor_field.embeddings_field_name:
                try:
                    marqo_document[constants.MARQO_DOC_TENSORS][tensor_field.name][
                        constants.MARQO_DOC_EMBEDDINGS] = list(field_content['blocks'].values())
                except (KeyError, AttributeError, TypeError) as e:
                    raise VespaDocumentParsingError(
                        f'Cannot parse embeddings field {field_name} with value {field_content}'
                    ) from e

        if self.fixed_fields.vespa_multimodal_params:
            marqo_document[common.MARQO_DOC_MULTIMODAL_PARAMS] = dict()
            for multimodal_field_name, serialized_multimodal_params in self.fixed_fields.vespa_multimodal_params.items():
                marqo_document[common.MARQO_DOC_MULTIMODAL_PARAMS][multimodal_field_name] = \
                    json.loads(serialized_multimodal_params)

        # Hybrid search raw scores
        if self.raw_tensor_score is not None:
            marqo_document[index_constants.MARQO_DOC_HYBRID_TENSOR_SCORE] = self.raw_tensor_score
        if self.raw_lexical_score is not None:
            marqo_document[index_constants.MARQO_DOC_HYBRID_LEXICAL_SCORE] = self.raw_lexical_score

        return marqo_document

    @classmethod
    def _verify_marqo_tensor_field_name(cls, field_name: str, marqo_index: SemiStructuredMarqoIndex):
        tensor_field_map = marqo_index.tensor_field_map
        if field_name not in tensor_field_map:
            raise InvalidFieldNameError(f'Invalid tensor field name {field_name} for index {marqo_index.name}. '
                                        f'Valid tensor field names are {", ".join(tensor_field_map.keys())}')

    @classmethod
    def _verify_marqo_tensor_field(cls, field_name: str, field_value: Dict[str, Any]):
        if not set(field_value.keys()) == {constants.MARQO_DOC_CHUNKS, constants.MARQO_DOC_EMBEDDINGS}:
            raise InvalidTensorFieldError(f'Invalid tensor field {field_name}. '
                                          f'Expected keys {constants.MARQO_DOC_CHUNKS}, {constants.MARQO_DOC_EMBEDDINGS} '
                                          f'but found {", ".join(field_value.keys())}')