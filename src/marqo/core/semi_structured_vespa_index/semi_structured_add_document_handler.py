from typing import Dict, Any

from marqo.core import constants
from marqo.core.constants import MARQO_DOC_ID
from marqo.core.document.models.add_docs_params import AddDocsParams
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex, Field, FieldType, FieldFeature, TensorField
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_index import SemiStructuredVespaIndex
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
from marqo.core.unstructured_vespa_index.unstructured_add_document_handler import UnstructuredAddDocumentsHandler
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.vespa.models import VespaDocument
from marqo.vespa.vespa_client import VespaClient


class SemiStructuredAddDocumentsHandler(UnstructuredAddDocumentsHandler):
    def __init__(self, marqo_index: SemiStructuredMarqoIndex, add_docs_params: AddDocsParams,
                 vespa_client: VespaClient, index_management: IndexManagement):
        super().__init__(marqo_index, add_docs_params, vespa_client)
        self.index_management = index_management
        self.marqo_index = marqo_index
        self.vespa_index = SemiStructuredVespaIndex(marqo_index)
        self.should_update_index = False

    def handle_field(self, marqo_doc, field_name, field_content):
        self._validate_field(field_name, field_content)
        text_field_type = self._infer_field_type(field_content)
        content = self.tensor_fields_container.collect(marqo_doc[MARQO_DOC_ID], field_name,
                                                       field_content, text_field_type)
        marqo_doc[field_name] = content

        if isinstance(content, str):
            # Add missing lexical fields to marqo index
            if field_name not in self.marqo_index.field_map:
                self.marqo_index.lexical_fields.append(
                    Field(name=field_name, type=FieldType.Text,
                          features=[FieldFeature.LexicalSearch],
                          lexical_field_name=f'{SemiStructuredVespaSchema.FIELD_INDEX_PREFIX}{field_name}')
                )
                self.marqo_index.clear_cache()
                self.should_update_index = True

    def to_vespa_doc(self, doc: Dict[str, Any]) -> VespaDocument:
        doc_tensor_fields = self.tensor_fields_container.get_tensor_field_content(doc[MARQO_DOC_ID])
        processed_tensor_fields = dict()
        for field_name, tensor_field_content in doc_tensor_fields.items():
            processed_tensor_fields[field_name] = {
                constants.MARQO_DOC_CHUNKS: tensor_field_content.tensor_field_chunks,
                constants.MARQO_DOC_EMBEDDINGS: tensor_field_content.tensor_field_embeddings,
            }
            # Add missing tensor fields to marqo index
            if field_name not in self.marqo_index.tensor_field_map:
                self.marqo_index.tensor_fields.append(TensorField(
                    name=field_name,
                    chunk_field_name=f'{SemiStructuredVespaSchema.FIELD_CHUNKS_PREFIX}{field_name}',
                    embeddings_field_name=f'{SemiStructuredVespaSchema.FIELD_EMBEDDING_PREFIX}{field_name}',
                ))
                self.marqo_index.clear_cache()
                self.should_update_index = True
        if processed_tensor_fields:
            doc[constants.MARQO_DOC_TENSORS] = processed_tensor_fields

        return VespaDocument(**self.vespa_index.to_vespa_document(marqo_document=doc))

    def pre_persist_to_vespa(self):
        if self.should_update_index:
            with RequestMetricsStore.for_request().time("add_documents.update_index"):
                self.index_management.update_index(self.marqo_index)

