import json
from typing import Dict, Any, List

import semver

from marqo import marqo_docs
from marqo.api import exceptions as api_errors
from marqo.core import constants
from marqo.core.constants import MARQO_DOC_ID
from marqo.core.inference.api import Modality, MediaDownloadError, Inference
from marqo.core.inference.modality_utils import infer_modality
from marqo.core.models import UnstructuredMarqoIndex
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import FieldType
from marqo.core.unstructured_vespa_index.common import MARQO_DOC_MULTIMODAL_PARAMS
from marqo.core.unstructured_vespa_index.unstructured_validation import validate_tensor_fields, validate_field_name, \
    validate_mappings_object_format, validate_coupling_of_mappings_and_doc
from marqo.core.unstructured_vespa_index.unstructured_vespa_index import UnstructuredVespaIndex
from marqo.core.vespa_index.add_documents_handler import AddDocumentsHandler, AddDocumentsError
from marqo.core.inference.tensor_fields_container import TensorFieldsContainer, TensorField

# TODO deps to tensor_search needs to be removed
from marqo.tensor_search.constants import ALLOWED_UNSTRUCTURED_FIELD_TYPES
from marqo.tensor_search.validation import validate_custom_vector, \
    validate_map_numeric_field
from marqo.vespa.models import VespaDocument
from marqo.vespa.models.get_document_response import Document
from marqo.vespa.vespa_client import VespaClient


class UnstructuredAddDocumentsHandler(AddDocumentsHandler):
    _MINIMUM_MARQO_VERSION_SUPPORTS_MAP_NUMERIC_FIELDS = semver.VersionInfo.parse("2.9.0")

    def __init__(self, marqo_index: UnstructuredMarqoIndex, add_docs_params: AddDocsParams, vespa_client: VespaClient,
                 inference: Inference):
        self._validate_add_docs_params(add_docs_params)
        super().__init__(marqo_index, add_docs_params, vespa_client, inference)
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
        content = self.tensor_fields_container.collect(marqo_doc[MARQO_DOC_ID], field_name, field_content)
        marqo_doc[field_name] = content

    def _infer_modality(self, tensor_field: TensorField) -> Modality:
        """Infer the modality of a tensor field. This is used for both unstructured and semi-structured indexes.

        We only infer the field type if the index is configured to treat URLs and pointers as images or media.

        treatUrlsAndPointersAsMedia is a new parameter introduced in Marqo 2.12 to support the new modalities
        of video and audio. Here is how it interacts with treatUrlsAndPointersAsImages:
        - Both False: All content is processed as text only
        - treatUrlsAndPointersAsImages True, treatUrlsAndPointersAsMedia False: Processes URLs and pointers as images.
        Other media types (video, audio) are treated as text
        - treatUrlsAndPointersAsImages False, treatUrlsAndPointersAsMedia True: Invalid state since this is a conflict
        - Both True: Processes URLs and pointers as various media types (images, videos, audio)

        The values of treatUrlsAndPointersAsMedia and treatUrlsAndPointersAsImages are validated in the MarqoIndex
        class, so we do not need to validate them here.

        Args:
            tensor_field: The tensor field to infer the modality.
        Returns:
            The inferred modality
        Raises:
            AddDocumentsError: If the modality of the media content cannot be inferred.
        """
        if (not self.marqo_index.treat_urls_and_pointers_as_images and
                not self.marqo_index.treat_urls_and_pointers_as_media):
            return Modality.TEXT

        try:
            modality = infer_modality(tensor_field.field_content, self.add_docs_params.media_download_headers)
        except MediaDownloadError as err:
            raise AddDocumentsError(f"Error processing {tensor_field.field_name}: {err.message}") from err

        if not self.marqo_index.treat_urls_and_pointers_as_media and modality in {Modality.AUDIO, Modality.VIDEO}:
            return Modality.TEXT

        return modality

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

            if isinstance(field_content, list):
                for element in field_content:
                    if not isinstance(element, str):
                        # if the field content is a list, it should only contain strings.
                        raise AddDocumentsError(
                            f"Field content {field_content} includes an element of type {type(element).__name__} "
                            f"which is not a string. Unstructured Marqo index only supports string lists."
                        )

            is_tensor_field = field_name in self.add_docs_params.tensor_fields

            if isinstance(field_content, dict):
                if self.tensor_fields_container.is_custom_tensor_field(field_name):
                    # TODO should is_non_tensor_field check be moved out to AddDocsParams validation?
                    # Please note that if one of the documents in the batch has a custom field which does not exist
                    # in the tensor field, the whole batch will fail and user will get a 400 pydantic.ValidationError.
                    # We keep this behaviour unchanged to be compatible with the legacy unstructured index.
                    validate_custom_vector(field_content, not is_tensor_field, self.marqo_index.model.get_dimension())
                elif self.marqo_index.parsed_marqo_version() < self._MINIMUM_MARQO_VERSION_SUPPORTS_MAP_NUMERIC_FIELDS:
                    # We do not support map of numeric fields prior to 2.9.0
                    raise AddDocumentsError(
                        f"The field {field_name} is a map field and only supported for indexes created with Marqo 2.9.0"
                        f" or later. See {marqo_docs.map_fields()} and {marqo_docs.mappings()}."
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
