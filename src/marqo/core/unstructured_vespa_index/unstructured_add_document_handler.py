import json
from typing import Dict, Any, Optional, List

import semver

from marqo import marqo_docs
from marqo.api import exceptions as api_errors
from marqo.core import constants
from marqo.core.constants import MARQO_DOC_ID
from marqo.core.vespa_index.add_documents_handler import AddDocumentsHandler, AddDocumentsError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.inference.tensor_fields_container import TensorFieldsContainer, MODALITY_FIELD_TYPE_MAP
from marqo.core.models import UnstructuredMarqoIndex
from marqo.core.models.marqo_index import FieldType
from marqo.core.unstructured_vespa_index.common import MARQO_DOC_MULTIMODAL_PARAMS
from marqo.core.unstructured_vespa_index.unstructured_validation import validate_tensor_fields, validate_field_name, \
    validate_mappings_object_format, validate_coupling_of_mappings_and_doc
from marqo.core.unstructured_vespa_index.unstructured_vespa_index import UnstructuredVespaIndex
from marqo.s2_inference.errors import MediaDownloadError
from marqo.s2_inference.multimodal_model_load import infer_modality, Modality

from marqo.vespa.models import VespaDocument
from marqo.vespa.models.get_document_response import Document

# TODO deps to tensor_search needs to be removed
from marqo.tensor_search.constants import ALLOWED_UNSTRUCTURED_FIELD_TYPES
from marqo.tensor_search.validation import list_types_valid, validate_custom_vector, \
    validate_multimodal_combination, validate_map_numeric_field
from marqo.vespa.vespa_client import VespaClient


class UnstructuredAddDocumentsHandler(AddDocumentsHandler):
    def __init__(self, marqo_index: UnstructuredMarqoIndex, add_docs_params: AddDocsParams, vespa_client: VespaClient):
        self._validate_add_docs_params(add_docs_params)
        super().__init__(marqo_index, add_docs_params, vespa_client)
        self.marqo_index = marqo_index
        self.vespa_index = UnstructuredVespaIndex(marqo_index)

    def _validate_add_docs_params(self, add_docs_params):
        validate_tensor_fields(add_docs_params.tensor_fields)
        if add_docs_params.mappings:
            validate_mappings_object_format(add_docs_params.mappings)

    def _create_tensor_fields_container(self) -> TensorFieldsContainer:
        mappings = self.add_docs_params.mappings or dict()
        return TensorFieldsContainer(
            tensor_fields=self.add_docs_params.tensor_fields,
            custom_vector_fields=[field_name for field_name, mapping in mappings.items()
                                  if mapping.get("type", None) == FieldType.CustomVector],
            multimodal_combo_fields={field_name: mapping['weights'] for field_name, mapping in mappings.items()
                                     if mapping.get("type", None) == FieldType.MultimodalCombination},
            should_normalise_custom_vector=self.should_normalise_custom_vector
        )

    def _validate_doc(self, doc):
        super()._validate_doc(doc)
        multimodal_sub_fields = list(self.tensor_fields_container.get_multimodal_sub_fields())
        if self.add_docs_params.mappings and multimodal_sub_fields:
            try:
                validate_coupling_of_mappings_and_doc(
                    doc, self.add_docs_params.mappings, multimodal_sub_fields
                )
            except api_errors.InvalidArgError as err:
                raise AddDocumentsError(err.message, error_code=err.code, status_code=err.status_code) from err

    def _handle_field(self, marqo_doc, field_name, field_content):
        self._validate_field(field_name, field_content)
        text_field_type = self._infer_field_type(field_content)
        content = self.tensor_fields_container.collect(marqo_doc[MARQO_DOC_ID], field_name,
                                                       field_content, text_field_type)
        marqo_doc[field_name] = content

    def _infer_field_type(self, field_content: Any) -> Optional[FieldType]:
        if not isinstance(field_content, str):
            return None

        try:
            modality = infer_modality(field_content)

            if not self.marqo_index.treat_urls_and_pointers_as_media and modality in [Modality.AUDIO, Modality.VIDEO]:
                modality = Modality.TEXT

            if not self.marqo_index.treat_urls_and_pointers_as_images and modality == Modality.IMAGE:
                modality = Modality.TEXT

            return MODALITY_FIELD_TYPE_MAP[modality]
        except MediaDownloadError as err:
            raise AddDocumentsError(err.message) from err

    def _validate_field(self, field_name: str, field_content: Any) -> None:
        try:
            # TODO extract the validation logic somewhere else
            validate_field_name(field_name)

            if type(field_content) not in ALLOWED_UNSTRUCTURED_FIELD_TYPES:
                raise AddDocumentsError(
                    f"Field content `{field_content}` \n"
                    f"of type `{type(field_content).__name__}` is not of valid content type!"
                    f"Allowed content types: {[ty.__name__ for ty in ALLOWED_UNSTRUCTURED_FIELD_TYPES]}"
                )

            if isinstance(field_content, list) and not list_types_valid(field_content):
                raise AddDocumentsError(
                    f"Field content '{field_content}' "
                    f"of type {type(field_content).__name__} is not of valid content type! "
                    f"All list elements must be of the same type and that type must be int, float or string"
                )

            if isinstance(field_content, dict):
                if self.tensor_fields_container.is_custom_tensor_field(field_name):
                    # TODO is_non_tensor_field check can be move out to AddDocsParams level
                    validate_custom_vector(field_content, False, self.marqo_index.model.get_dimension())
                elif self.tensor_fields_container.is_multimodal_field(field_name):
                    # FIXME, multimodal field should not be present in the doc
                    # TODO This validation should be done at AddDocsParams level
                    validate_multimodal_combination(field_content, False,
                                                    self.tensor_fields_container.get_multimodal_field_mapping(field_name))
                elif self.marqo_index.parsed_marqo_version() < semver.VersionInfo.parse("2.9.0"):
                    # TODO This check should not happen at root level
                    raise AddDocumentsError(
                        f"The field {field_name} is a map field and only supported for indexes created with Marqo 2.9.0 "
                        f"or later. See {marqo_docs.map_fields()} and {marqo_docs.mappings()}."
                    )
                else:
                    validate_map_numeric_field(field_content)
        except (api_errors.InvalidFieldNameError, api_errors.InvalidArgError) as err:
            raise AddDocumentsError(err.message, error_code=err.code, status_code=err.status_code) from err

    def _handle_multi_modal_fields(self, marqo_doc: Dict[str, Any]) -> None:
        doc_id = marqo_doc[MARQO_DOC_ID]
        for field_name, weights in self.tensor_fields_container.collect_multi_modal_fields(
                doc_id, self.marqo_index.normalize_embeddings):

            if MARQO_DOC_MULTIMODAL_PARAMS not in marqo_doc:
                marqo_doc[MARQO_DOC_MULTIMODAL_PARAMS] = dict()

            marqo_doc[MARQO_DOC_MULTIMODAL_PARAMS][field_name] = json.dumps({
                'weights': weights,
                'type': FieldType.MultimodalCombination
            })

    def _populate_existing_tensors(self, existing_vespa_docs: List[Document]):
        if not self.add_docs_params.use_existing_tensors or not existing_vespa_docs:
            return

        for vespa_doc in existing_vespa_docs:
            existing_marqo_doc = self.vespa_index.to_marqo_document(vespa_doc.dict())
            existing_multimodal_weights = {
                field_name: mapping['weights']
                for field_name, mapping in existing_marqo_doc.get(MARQO_DOC_MULTIMODAL_PARAMS, dict()).items()
            }
            self.tensor_fields_container.populate_tensor_from_existing_doc(existing_marqo_doc,
                                                                           existing_multimodal_weights)

    def _to_vespa_doc(self, doc: Dict[str, Any]) -> VespaDocument:
        all_chunks = []
        all_embeddings = []
        doc_tensor_fields = self.tensor_fields_container.get_tensor_field_content(doc[MARQO_DOC_ID])
        for field_name, tensor_field_content in doc_tensor_fields.items():
            all_chunks.extend([f'{field_name}::{chunk}' for chunk in tensor_field_content.tensor_field_chunks])
            all_embeddings.extend(tensor_field_content.tensor_field_embeddings)
        doc[constants.MARQO_DOC_CHUNKS] = all_chunks
        doc[constants.MARQO_DOC_EMBEDDINGS] = {index: embedding for index, embedding in enumerate(all_embeddings)}
        return VespaDocument(**self.vespa_index.to_vespa_document(marqo_document=doc))
