import json
from contextlib import ExitStack
from typing import Dict, Any, List, Optional, Tuple, Callable, Protocol

import numpy as np
import semver

from marqo import marqo_docs
from marqo.config import Config
from marqo.core import constants
from marqo.core.constants import MARQO_DOC_ID
from marqo.core.document.add_documents_handler import AddDocumentsHandler, AddDocumentsError
from marqo.core.document.models.add_docs_params import AddDocsParams
from marqo.core.document.tensor_fields_container import TensorFieldsContainer, Chunker, Vectoriser
from marqo.core.models import UnstructuredMarqoIndex
from marqo.core.models.marqo_index import FieldType
from marqo.core.unstructured_vespa_index.common import MARQO_DOC_MULTIMODAL_PARAMS
from marqo.core.unstructured_vespa_index.unstructured_validation import validate_tensor_fields, validate_field_name, \
    validate_mappings_object_format, validate_coupling_of_mappings_and_doc
from marqo.core.unstructured_vespa_index.unstructured_vespa_index import UnstructuredVespaIndex
from marqo.s2_inference.processing import image as image_processor
from marqo.s2_inference.processing import text as text_processor
from marqo.s2_inference import s2_inference
from marqo.tensor_search import add_docs

# TODO deps to tensor_search needs to be removed
from marqo.tensor_search.constants import ALLOWED_UNSTRUCTURED_FIELD_TYPES
from marqo.tensor_search.validation import list_types_valid, validate_custom_vector, \
    validate_multimodal_combination, validate_map_numeric_field
from marqo.vespa.models import VespaDocument
from marqo.vespa.models.get_document_response import Document
from marqo.api import exceptions as api_errors
from marqo.s2_inference import errors as s2_inference_errors


class UnstructuredAddDocumentsHandler(AddDocumentsHandler):
    def __init__(self, marqo_index: UnstructuredMarqoIndex, config: Config, add_docs_params: AddDocsParams):
        validate_tensor_fields(add_docs_params.tensor_fields)
        if add_docs_params.mappings:
            validate_mappings_object_format(add_docs_params.mappings)

        super().__init__(marqo_index, config, add_docs_params)

        self.tensor_fields_container = TensorFieldsContainer(
            add_docs_params.tensor_fields,
            add_docs_params.mappings or dict(),
            marqo_index.treat_urls_and_pointers_as_images
        )

        self.vespa_index = UnstructuredVespaIndex(marqo_index)

    def validate_doc(self, doc):
        super().validate_doc(doc)
        multimodal_sub_fields = list(self.tensor_fields_container.get_multimodal_sub_fields())
        if self.add_docs_params.mappings and multimodal_sub_fields:
            try:
                validate_coupling_of_mappings_and_doc(
                    doc, self.add_docs_params.mappings, multimodal_sub_fields
                )
            except api_errors.InvalidArgError as err:
                raise AddDocumentsError(err.message, error_code=err.code, status_code=err.status_code) from err

    def handle_field(self, marqo_doc, field_name, field_content):
        self._validate_field(field_name, field_content)
        content = self.tensor_fields_container.collect(marqo_doc[MARQO_DOC_ID], field_name, field_content)
        marqo_doc[field_name] = content

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

    def _download_image_contents(self, exit_stack):
        # collect image urls
        # consider collect these info while building tensor_fields_container
        url_doc_id_map = dict()
        doc_image_fields = dict()
        image_tensor_fields = set()
        for doc_id, field_name, tensor_field_content in (
                self.tensor_fields_container.tensor_fields_to_vectorise(FieldType.ImagePointer)):
            url = tensor_field_content.field_content

            if url not in url_doc_id_map:
                url_doc_id_map[url] = set()
            url_doc_id_map[url].add(doc_id)

            if doc_id not in doc_image_fields:
                doc_image_fields[doc_id] = dict()
            doc_image_fields[doc_id][field_name] = url

            image_tensor_fields.add(field_name)

        if not doc_image_fields:
            return dict()

        # TODO refactor download_and_preprocess_images to accept dict(doc_id, list(urls))
        image_repo = exit_stack.enter_context(
            add_docs.download_and_preprocess_images(
                docs=list(doc_image_fields.values()),
                thread_count=self.add_docs_params.image_download_thread_count,
                tensor_fields=image_tensor_fields,
                image_download_headers=self.add_docs_params.image_download_headers,
                model_name=self.marqo_index.model.name,
                normalize_embeddings=self.marqo_index.normalize_embeddings,
                model_properties=self.marqo_index.model.get_properties(),
                device=self.add_docs_params.device,
                model_auth=self.add_docs_params.model_auth,
                patch_method_exists=self.marqo_index.image_preprocessing.patch_method is not None
            )
        )

        for url, data in image_repo.items():
            if isinstance(data, Exception):
                for doc_id in url_doc_id_map[url]:
                    self.add_docs_response_collector.collect_error_response(doc_id, AddDocumentsError(
                        error_message=f"Could not find image found at `{url}`. Reason: {str(data)}"
                    ))
                    self.tensor_fields_container.remove_doc(doc_id)

        return image_repo

    def vectorise_tensor_fields(self) -> None:
        with ExitStack() as exit_stack:
            chunkers: Dict[FieldType, Chunker] = {
                FieldType.Text: self.text_chunker(),
                FieldType.ImagePointer: self.image_chunker(exit_stack)
            }
            vectoriser = self.vectoriser()

            for doc_id, field_name, tensor_field_content in (
                    self.tensor_fields_container.tensor_fields_to_vectorise(*chunkers.keys())):
                try:
                    tensor_field_content.chunk(chunkers)
                    tensor_field_content.vectorise(vectoriser)
                except AddDocumentsError as err:
                    #  TODO make sure the chunkers and vectoriser all throws AddDocumentError
                    self.add_docs_response_collector.collect_error_response(doc_id, err)
                    self.tensor_fields_container.remove_doc(doc_id)

    def text_chunker(self) -> Chunker:
        text_chunk_prefix = self.marqo_index.model.get_text_chunk_prefix(self.add_docs_params.text_chunk_prefix)
        text_preprocessing = self.marqo_index.text_preprocessing

        def chunk(field_content: str, single_chunk: bool = False) -> Tuple[List[str], List[str]]:
            chunks = [field_content] if single_chunk else (
                text_processor.split_text(text=field_content,
                                          split_by=text_preprocessing.split_method.value,
                                          split_length=text_preprocessing.split_length,
                                          split_overlap=text_preprocessing.split_overlap))
            return chunks, text_processor.prefix_text_chunks(chunks, text_chunk_prefix)

        return chunk

    def image_chunker(self, exit_stack) -> Chunker:
        image_repo = self._download_image_contents(exit_stack)
        image_method = self.marqo_index.image_preprocessing.patch_method

        def chunk(field_content: str, single_chunk: bool = False):
            url = field_content
            image_data = image_repo[url]
            if single_chunk or image_method is None:
                return [url], [image_data]

            try:
                content_chunks, text_chunks = image_processor.chunk_image(
                    image_data, device=self.add_docs_params.device, method=image_method.value)
                return text_chunks, content_chunks
            except s2_inference_errors.S2InferenceError as e:
                raise AddDocumentsError(e.message)

        return chunk

    def vectoriser(self) -> Vectoriser:
        def vectorise(content_chunks: List[str], field_type: FieldType):
            # TODO batch request to get more GPU utilisation, also consider how to fail fast
            try:
                return s2_inference.vectorise(
                    model_name=self.marqo_index.model.name,
                    model_properties=self.marqo_index.model.get_properties(),
                    content=content_chunks,
                    device=self.add_docs_params.device,
                    normalize_embeddings=self.marqo_index.normalize_embeddings,
                    infer=field_type == FieldType.ImagePointer,
                    model_auth=self.add_docs_params.model_auth
                )
            except (s2_inference_errors.UnknownModelError,
                    s2_inference_errors.InvalidModelPropertiesError,
                    s2_inference_errors.ModelLoadError,
                    s2_inference.ModelDownloadError) as model_error:
                # Fail the whole batch due to a malfunctioning embedding model
                # TODO Add a core exception
                raise api_errors.BadRequestError(
                    message=f'Problem vectorising query. Reason: {str(model_error)}',
                    link=marqo_docs.list_of_models()
                )
            except s2_inference_errors.S2InferenceError as e:
                raise AddDocumentsError(e.message)

        return vectorise

    def persist_to_vespa(self) -> None:
        vespa_docs = []
        for doc_id, doc in self.add_docs_response_collector.marqo_docs.items():
            all_chunks = []
            all_embeddings = []
            for field_name, tensor_field_content in self.tensor_fields_container.get_tensor_field_content(doc_id).items():
                all_chunks.extend([f'{field_name}::{chunk}' for chunk in tensor_field_content.tensor_field_chunks])
                all_embeddings.extend(tensor_field_content.tensor_field_embeddings)
            doc[constants.MARQO_DOC_CHUNKS] = all_chunks
            doc[constants.MARQO_DOC_EMBEDDINGS] = {index: embedding for index, embedding in enumerate(all_embeddings)}

            vespa_docs.append(VespaDocument(**self.vespa_index.to_vespa_document(marqo_document=doc)))

        index_responses = self.config.vespa_client.feed_batch(vespa_docs, self.marqo_index.schema_name)

        for resp in index_responses.responses:
            # FIXME doc_id is not url encoded
            doc_id = resp.id.split('::')[-1] if resp.id else None
            status, message = self.config.document.translate_vespa_document_response(resp.status, message=resp.message)
            if status != 200:
                self.add_docs_response_collector.collect_error_response(doc_id, AddDocumentsError(
                    error_message=resp.message, status_code=resp.status, error_code='vespa_error'  # breaking?
                ))
            else:
                self.add_docs_response_collector.collect_successful_response(doc_id)

