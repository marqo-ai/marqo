import json
from typing import Dict, Any

from marqo.api import exceptions as api_errors
from marqo.config import Config
from marqo.core import constants
from marqo.core.constants import MARQO_DOC_ID
from marqo.core.document.add_documents_handler import AddDocumentsHandler, AddDocumentsError
from marqo.core.document.models.add_docs_params import AddDocsParams
from marqo.core.document.tensor_fields_container import TensorFieldsContainer
from marqo.core.models.marqo_index import FieldType, StructuredMarqoIndex
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
from marqo.core.unstructured_vespa_index.common import MARQO_DOC_MULTIMODAL_PARAMS
# TODO deps to tensor_search needs to be removed
from marqo.tensor_search import validation
from marqo.vespa.models import VespaDocument
from marqo.vespa.models.get_document_response import Document


class StructuredAddDocumentsHandler(AddDocumentsHandler):
    def __init__(self, marqo_index: StructuredMarqoIndex, config: Config, add_docs_params: AddDocsParams):
        self._validate_add_docs_params(add_docs_params, marqo_index)
        super().__init__(marqo_index, config, add_docs_params)
        self.marqo_index = marqo_index
        self.vespa_index = StructuredVespaIndex(marqo_index)

    def _create_tensor_fields_container(self) -> TensorFieldsContainer:
        # TODO change this
        return TensorFieldsContainer(
            list(self.marqo_index.tensor_field_map.keys()),
            self.add_docs_params.mappings or dict()
        )

    def _validate_add_docs_params(self, add_docs_params: AddDocsParams, marqo_index: StructuredMarqoIndex):
        if add_docs_params.tensor_fields is not None:
            raise api_errors.InvalidArgError("Cannot specify 'tensorFields' when adding documents to a "
                                             "structured index. 'tensorFields' must be defined in structured "
                                             "index schema at index creation time")

        # TODO confirm if the customer vector mapping is allowed for structured index. I guess it's ignored
        if add_docs_params.mappings is not None:
            validation.validate_mappings_object(add_docs_params.mappings, marqo_index)

    def handle_field(self, marqo_doc, field_name, field_content):
        self._validate_field(field_name, field_content)
        content = self.tensor_fields_container.collect(marqo_doc[MARQO_DOC_ID], field_name, field_content)
        marqo_doc[field_name] = content

    def _validate_field(self, field_name: str, field_content: Any) -> None:
        try:
            # TODO extract the validation logic somewhere else
            validation.validate_field_name(field_name)

            if field_name not in self.marqo_index.field_map:
                raise AddDocumentsError(f"Field {field_name} is not a valid field for structured index "
                                        f"{self.marqo_index.name}. Valid fields are: "
                                        f"{', '.join(self.marqo_index.field_map.keys())}")

            field_type = self.marqo_index.field_map[field_name].type
            if field_type == FieldType.MultimodalCombination:
                raise AddDocumentsError(f"Field {field_name} is a multimodal combination field and cannot "
                                        f"be assigned a value.")

            # FIXME, also include subfields
            is_tensor_field = field_name in self.marqo_index.tensor_field_map
            validation.validate_field_content(
                field_content=field_content,
                is_non_tensor_field=not is_tensor_field
            )
            # Used to validate custom_vector field or any other new dict field type
            # TODO inline and remove unstructured validation logic
            if isinstance(field_content, dict):
                validation.validate_dict(
                    field=field_name, field_content=field_content,
                    is_non_tensor_field=not is_tensor_field,
                    mappings=self.add_docs_params.mappings,
                    index_model_dimensions=self.marqo_index.model.get_dimension(),
                    structured_field_type=field_type,
                    marqo_index_version=self.marqo_index.parsed_marqo_version())
        except (api_errors.InvalidFieldNameError, api_errors.InvalidArgError) as err:
            raise AddDocumentsError(err.message, error_code=err.code, status_code=err.status_code) from err

    def handle_multi_modal_fields(self, marqo_doc: Dict[str, Any]) -> None:
        doc_id = marqo_doc[MARQO_DOC_ID]
        for field_name, mapping in self.tensor_fields_container.collect_multi_modal_fields(
                doc_id, self.marqo_index.normalize_embeddings):

            if MARQO_DOC_MULTIMODAL_PARAMS not in marqo_doc:
                marqo_doc[MARQO_DOC_MULTIMODAL_PARAMS] = dict()
            marqo_doc[MARQO_DOC_MULTIMODAL_PARAMS][field_name] = json.dumps(mapping)

    def handle_existing_tensors(self, existing_vespa_docs: Dict[str, Document]):
        if not self.add_docs_params.use_existing_tensors or not existing_vespa_docs:
            return

        for doc_id, vespa_doc in existing_vespa_docs.items():
            existing_marqo_doc = self.vespa_index.to_marqo_document(vespa_doc.dict())
            existing_multimodal_mappings = existing_marqo_doc.get(MARQO_DOC_MULTIMODAL_PARAMS, dict())
            self.tensor_fields_container.populate_tensor_from_existing_doc(doc_id, existing_marqo_doc,
                                                                           existing_multimodal_mappings)

    def to_vespa_doc(self, doc: Dict[str, Any]) -> VespaDocument:
        doc_tensor_fields = self.tensor_fields_container.get_tensor_field_content(doc[MARQO_DOC_ID])
        processed_tensor_fields = dict()
        for field_name, tensor_field_content in doc_tensor_fields.items():
            processed_tensor_fields[field_name] = {
                constants.MARQO_DOC_CHUNKS: tensor_field_content.chunks,
                constants.MARQO_DOC_EMBEDDINGS: tensor_field_content.embeddings,
            }
        if processed_tensor_fields:
            doc[constants.MARQO_DOC_TENSORS] = processed_tensor_fields

        return VespaDocument(**self.vespa_index.to_vespa_document(marqo_document=doc))
