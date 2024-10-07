from typing import Dict, Any

import pydantic

from marqo.base_model import ImmutableStrictBaseModel
from marqo.core import constants
from marqo.core.constants import MARQO_DOC_ID
from marqo.core.exceptions import TooManyFieldsError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex, Field, FieldType, FieldFeature, TensorField
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
from marqo.core.unstructured_vespa_index.unstructured_add_document_handler import UnstructuredAddDocumentsHandler
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


class SemiStructuredAddDocumentsHandler(UnstructuredAddDocumentsHandler):
    def __init__(self, marqo_index: SemiStructuredMarqoIndex, add_docs_params: AddDocsParams,
                 vespa_client: VespaClient, index_management: IndexManagement,
                 field_count_config=SemiStructuredFieldCountConfig()):
        super().__init__(marqo_index, add_docs_params, vespa_client)
        self.index_management = index_management
        self.marqo_index = marqo_index
        self.vespa_index = SemiStructuredVespaIndex(marqo_index)
        self.should_update_index = False
        self.field_count_config = field_count_config

    def _handle_field(self, marqo_doc, field_name, field_content):
        self._validate_field(field_name, field_content)
        text_field_type = self._infer_field_type(field_content)
        content = self.tensor_fields_container.collect(marqo_doc[MARQO_DOC_ID], field_name,
                                                       field_content, text_field_type)
        marqo_doc[field_name] = content

        if isinstance(content, str):
            self._add_lexical_field_to_index(field_name)

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
        self.marqo_index.lexical_fields.append(
            Field(name=field_name, type=FieldType.Text,
                  features=[FieldFeature.LexicalSearch],
                  lexical_field_name=f'{SemiStructuredVespaSchema.FIELD_INDEX_PREFIX}{field_name}')
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
        if field_name not in self.marqo_index.tensor_field_map:
            self.marqo_index.tensor_fields.append(TensorField(
                name=field_name,
                chunk_field_name=f'{SemiStructuredVespaSchema.FIELD_CHUNKS_PREFIX}{field_name}',
                embeddings_field_name=f'{SemiStructuredVespaSchema.FIELD_EMBEDDING_PREFIX}{field_name}',
            ))
            self.marqo_index.clear_cache()
            self.should_update_index = True

