from typing import Dict, Any

import pydantic

from marqo.base_model import ImmutableStrictBaseModel
from marqo.core import constants
from marqo.core.constants import MARQO_DOC_ID
from marqo.core.exceptions import TooManyFieldsError
from marqo.core.inference.api import Inference
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex, Field, FieldType, FieldFeature, TensorField, \
    StringArrayField
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
from marqo.core.unstructured_vespa_index.unstructured_add_document_handler import UnstructuredAddDocumentsHandler
from marqo.core.vespa_index.add_documents_handler import logger
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.tensor_search.utils import read_env_vars_and_defaults_ints
from marqo.vespa.models import VespaDocument
from marqo.vespa.vespa_client import VespaClient


class SemiStructuredFieldCountConfig(ImmutableStrictBaseModel):
    # TODO find a way to decouple from env vars when retrieving configurations
    max_lexical_field_count: int = pydantic.Field(default_factory=lambda: read_env_vars_and_defaults_ints(
        EnvVars.MARQO_MAX_LEXICAL_FIELD_COUNT_UNSTRUCTURED))
    max_tensor_field_count: int = pydantic.Field(default_factory=lambda: read_env_vars_and_defaults_ints(
        EnvVars.MARQO_MAX_TENSOR_FIELD_COUNT_UNSTRUCTURED))
    max_string_array_field_count: int = pydantic.Field(default_factory=lambda: read_env_vars_and_defaults_ints(
        EnvVars.MARQO_MAX_STRING_ARRAY_FIELD_COUNT_UNSTRUCTURED))


class SemiStructuredAddDocumentsHandler(UnstructuredAddDocumentsHandler):
    def __init__(self, marqo_index: SemiStructuredMarqoIndex, add_docs_params: AddDocsParams,
                 vespa_client: VespaClient, index_management: IndexManagement, inference: Inference,
                 field_count_config=SemiStructuredFieldCountConfig()):
        super().__init__(marqo_index, add_docs_params, vespa_client, inference)
        self.index_management = index_management
        self.marqo_index = marqo_index
        self.vespa_index = SemiStructuredVespaIndex(marqo_index)
        self.should_update_index = False
        self.field_count_config = field_count_config

    def _handle_field(self, marqo_doc, field_name, field_content):
        """Handle a field in a Marqo document by processing it and updating the index schema if needed.
        
        Args:
            marqo_doc: The Marqo document being processed
            field_name: Name of the field
            field_content: Content of the field
        """
        # Process field using parent class handler
        super()._handle_field(marqo_doc, field_name, field_content)

        # Add lexical field if content is a string
        if isinstance(marqo_doc[field_name], str):
            self._add_lexical_field_to_index(field_name)

        # Add string array field if content is list of strings and index version supports it
        is_string_array = (
            isinstance(field_content, list) and 
            all(isinstance(elem, str) for elem in field_content)
        )
        if (is_string_array and
                # This is required so that we can update schema on the fly
                self.marqo_index.index_supports_partial_updates):
            self._add_string_array_field_to_index(field_name)

    def _to_vespa_doc(self, doc: Dict[str, Any]) -> VespaDocument:
        doc_tensor_fields = self.tensor_fields_container.get_tensor_field_content(doc[MARQO_DOC_ID])
        processed_tensor_fields = dict()
        for field_name, tensor_field_content in doc_tensor_fields.items():
            processed_tensor_fields[field_name] = {
                constants.MARQO_DOC_CHUNKS: tensor_field_content.tensor_field_chunks,
                constants.MARQO_DOC_EMBEDDINGS: tensor_field_content.tensor_field_embeddings,
            }
            self._add_tensor_field_to_index(field_name)
        if processed_tensor_fields:
            doc[constants.MARQO_DOC_TENSORS] = processed_tensor_fields

        # doc here is Dict[str, Any], which will be converted to a VespaDocument
        return VespaDocument(**self.vespa_index.to_vespa_document(marqo_document=doc))

    def _pre_persist_to_vespa(self):
        if self.should_update_index:
            with RequestMetricsStore.for_request().time("add_documents.update_index"):
                self.index_management.update_index(self.marqo_index)
            # Force fresh this index in the index cache to make sure the following search requests get the latest index
            # TODO this is a temporary solution to fix the consistency issue for single instance Marqo (used extensively
            #   in api-tests and integration tests). Find a better way to solve consistency issue for Marqo clusters
            from marqo.tensor_search import index_meta_cache
            index_meta_cache.get_index(self.index_management, self.marqo_index.name, force_refresh=True)

    def _add_lexical_field_to_index(self, field_name):
        if field_name in self.marqo_index.field_map:
            return

        max_lexical_field_count = self.field_count_config.max_lexical_field_count
        if len(self.marqo_index.lexical_fields) >= max_lexical_field_count:
            raise TooManyFieldsError(f'Index {self.marqo_index.name} has {len(self.marqo_index.lexical_fields)} '
                                     f'lexical fields. Your request to add {field_name} as a lexical field is rejected '
                                     f'since it exceeds the limit of {max_lexical_field_count}. Please set a larger '
                                     f'limit in MARQO_MAX_LEXICAL_FIELD_COUNT_UNSTRUCTURED environment variable.')

        # Add missing lexical fields to marqo index
        logger.debug(f'Adding lexical field {field_name} to index {self.marqo_index.name}')

        self.marqo_index.lexical_fields.append(
            Field(name=field_name, type=FieldType.Text,
                  features=[FieldFeature.LexicalSearch],
                  lexical_field_name=f'{SemiStructuredVespaSchema.FIELD_INDEX_PREFIX}{field_name}')
        )
        self.marqo_index.clear_cache()
        self.should_update_index = True

    def _add_string_array_field_to_index(self, field_name):
        if field_name in self.marqo_index.name_to_string_array_field_map:
            return

        max_string_array_field_count = self.field_count_config.max_string_array_field_count
        if len(self.marqo_index.string_array_fields) >= max_string_array_field_count:
            raise TooManyFieldsError(f'Index {self.marqo_index.name} has {len(self.marqo_index.string_array_fields)} '
                                     f'string array fields. Your request to add {field_name} as a string array field is '
                                     f'rejected since it exceeds the limit of {max_string_array_field_count}. Please set '
                                     f'a larger limit in MARQO_MAX_STRING_ARRAY_FIELD_COUNT_UNSTRUCTURED environment variable.')

        logger.debug(f'Adding string array field {field_name} to index {self.marqo_index.name}')

        self.marqo_index.string_array_fields.append(
            StringArrayField(
                name=field_name, type=FieldType.ArrayText, features=[FieldFeature.Filter],
                string_array_field_name=f'{SemiStructuredVespaSchema.FIELD_STRING_ARRAY_PREFIX}{field_name}')
        )
        self.marqo_index.clear_cache()
        self.should_update_index = True

    def _add_tensor_field_to_index(self, field_name):
        if field_name in self.marqo_index.tensor_field_map:
            return

        max_tensor_field_count = self.field_count_config.max_tensor_field_count
        if len(self.marqo_index.tensor_fields) >= max_tensor_field_count:
            raise TooManyFieldsError(f'Index {self.marqo_index.name} has {len(self.marqo_index.tensor_fields)} '
                                     f'tensor fields. Your request to add {field_name} as a tensor field is rejected '
                                     f'since it exceeds the limit of {max_tensor_field_count}. Please set a larger '
                                     f'limit in MARQO_MAX_TENSOR_FIELD_COUNT_UNSTRUCTURED environment variable.')

        # Add missing tensor fields to marqo index
        logger.debug(f'Adding tensor field {field_name} to index {self.marqo_index.name}')

        if field_name not in self.marqo_index.tensor_field_map:
            self.marqo_index.tensor_fields.append(TensorField(
                name=field_name,
                chunk_field_name=f'{SemiStructuredVespaSchema.FIELD_CHUNKS_PREFIX}{field_name}',
                embeddings_field_name=f'{SemiStructuredVespaSchema.FIELD_EMBEDDING_PREFIX}{field_name}',
            ))
            self.marqo_index.clear_cache()
            self.should_update_index = True

