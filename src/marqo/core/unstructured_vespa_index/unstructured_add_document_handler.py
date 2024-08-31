from typing import Dict, Any, List

import semver

from marqo import marqo_docs
from marqo.api.exceptions import InvalidArgError
from marqo.config import Config
from marqo.core import constants
from marqo.core.document.add_documents_handler import AddDocumentsHandler
from marqo.core.document.models.add_docs_params import AddDocsParams
from marqo.core.document.tensor_fields_container import TensorFieldsContainer
from marqo.core.models import UnstructuredMarqoIndex
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsItem
from marqo.core.models.marqo_index import FieldType
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


class UnstructuredAddDocumentsHandler(AddDocumentsHandler):
    def __init__(self, marqo_index: UnstructuredMarqoIndex, config: Config, add_docs_params: AddDocsParams):
        validate_tensor_fields(add_docs_params.tensor_fields)
        validate_mappings_object_format(add_docs_params.mappings)

        super().__init__(marqo_index, config, add_docs_params)

        self.tensor_fields_container = TensorFieldsContainer(
            add_docs_params.tensor_fields,
            add_docs_params.mappings or dict(),
            marqo_index.treat_urls_and_pointers_as_images
        )

        self.vespa_index = UnstructuredVespaIndex(marqo_index)

    def validate_doc(self, doc, loc: int) -> None:
        super().validate_doc(doc, loc)
        multimodal_sub_fields = list(self.tensor_fields_container.get_multimodal_sub_fields())
        if self.add_docs_params.mappings and multimodal_sub_fields:
            validate_coupling_of_mappings_and_doc(
                doc, self.add_docs_params.mappings, multimodal_sub_fields
            )

    def handle_field(self, marqo_doc, field_name, field_content):
        self._validate_field(field_name, field_content)
        content = self.tensor_fields_container.collect(marqo_doc['_id'], field_name, field_content)
        marqo_doc[field_name] = content

    def _validate_field(self, field_name: str, field_content: Any) -> None:
        validate_field_name(field_name)

        if type(field_content) not in ALLOWED_UNSTRUCTURED_FIELD_TYPES:
            raise InvalidArgError(
                f"Field content `{field_content}` \n"
                f"of type `{type(field_content).__name__}` is not of valid content type!"
                f"Allowed content types: {[ty.__name__ for ty in ALLOWED_UNSTRUCTURED_FIELD_TYPES]}"
            )

        if isinstance(field_content, list) and not list_types_valid(field_content):
            raise InvalidArgError(
                f"Field content '{field_content}' "
                f"of type {type(field_content).__name__} is not of valid content type! "
                f"All list elements must be of the same type and that type must be int, float or string"
            )

        if isinstance(field_content, dict):
            if self.tensor_fields_container.is_custom_tensor_field(field_name):
                # TODO is_non_tensor_field check can be move out to AddDocsParams level
                validate_custom_vector(field_content, False, self.marqo_index.model.get_dimension())
            elif self.tensor_fields_container.is_multimodal_field(field_name):
                # TODO is_non_tensor_field check can be move out to AddDocsParams level
                validate_multimodal_combination(field_content, False,
                                                self.tensor_fields_container.get_multimodal_field_mapping(field_name))
            elif self.marqo_index.parsed_marqo_version() < semver.VersionInfo.parse("2.9.0"):
                # TODO better to have version check extract to a common place
                raise InvalidArgError(
                    f"The field {field_name} is a map field and only supported for indexes created with Marqo 2.9.0 "
                    f"or later. See {marqo_docs.map_fields()} and {marqo_docs.mappings()}."
                )
            else:
                validate_map_numeric_field(field_content)

    def handle_existing_tensors(self, existing_vespa_docs: Dict[str, Document]):
        if not self.add_docs_params.use_existing_tensors or not existing_vespa_docs:
            return

        for doc_id, vespa_doc in existing_vespa_docs.items():
            existing_marqo_doc = self.vespa_index.to_marqo_document(vespa_doc.dict())
            self.tensor_fields_container.populate_tensor_from_existing_doc(doc_id, existing_marqo_doc)

    def _download_and_chunk_image_contents(self, invalid_docs: List[MarqoAddDocumentsItem]) -> None:
        # download images
        url_field_map = dict()
        doc_url_map = dict()
        tensor_fields = set()
        for doc_id, field_name, tensor_field_content in (
                self.tensor_fields_container.tensor_fields_to_vectorise(FieldType.ImagePointer)):
            field_type, url = tensor_field_content.field_type, tensor_field_content.field_content

            if url not in url_field_map:
                url_field_map[url] = []
            url_field_map[url].append((doc_id, tensor_field_content))

            if doc_id not in doc_url_map:
                doc_url_map[doc_id] = dict()
            doc_url_map[doc_id][field_name] = url

            tensor_fields.add(field_name)

        if not doc_url_map:
            return

        # TODO refactor download_and_preprocess_images to accept dict(doc_id, list(urls))
        # FIXME maybe pass in add_docs_params and marqo index directly?
        with add_docs.download_and_preprocess_images(
            docs=doc_url_map.values(),
            thread_count=self.add_docs_params.image_download_thread_count,
            tensor_fields=tensor_fields,
            image_download_headers=self.add_docs_params.image_download_headers,
            model_name=self.marqo_index.model.name,
            normalize_embeddings=self.marqo_index.normalize_embeddings,
            model_properties=self.marqo_index.model.get_properties(),
            device=self.add_docs_params.device,
            model_auth=self.add_docs_params.model_auth,
            patch_method_exists=self.marqo_index.image_preprocessing.patch_method is not None
        ) as image_repo:
            image_method = self.marqo_index.image_preprocessing.patch_method
            for url, data in image_repo.items():
                if isinstance(data, Exception):
                    for doc_id in set([doc_id for (doc_id, _) in url_field_map[url]]):
                        invalid_docs.append(MarqoAddDocumentsItem.from_error(doc_id, data))
                        self.tensor_fields_container.remove_doc(doc_id)
                else:
                    if image_method is not None:
                        # TODO handle error
                        content_chunks, text_chunks = image_processor.chunk_image(
                            data, device=self.add_docs_params.device, method=image_method.value)
                    else:
                        content_chunks, text_chunks = [data], [url]

                    for (_, tensor_field_content) in url_field_map[url]:
                        tensor_field_content.chunks = text_chunks
                        tensor_field_content.content_chunks = content_chunks

    def vectorise_tensor_fields(self) -> List[MarqoAddDocumentsItem]:
        invalid_docs: List[MarqoAddDocumentsItem] = []
        self._chunk_text_content(invalid_docs)
        self._download_and_chunk_image_contents(invalid_docs)
        self._batch_vectorise(invalid_docs)
        # TODO handle multimodal fields

        return invalid_docs

    def persist_to_vespa(self, marqo_docs: Dict[int, Dict[str, Any]]) -> List[MarqoAddDocumentsItem]:
        vespa_docs = []
        for doc in marqo_docs.values():
            all_chunks = []
            all_embeddings = []
            for field_name, tensor_field_content in self.tensor_fields_container.get_tensor_field_content(doc['_id']).items():
                all_chunks.extend([f'{field_name}::{chunk}' for chunk in tensor_field_content.chunks])
                all_embeddings.extend(tensor_field_content.embeddings)
            doc[constants.MARQO_DOC_CHUNKS] = all_chunks
            doc[constants.MARQO_DOC_EMBEDDINGS] = {index: embedding for index, embedding in enumerate(all_embeddings)}

            vespa_docs.append(VespaDocument(**self.vespa_index.to_vespa_document(marqo_document=doc)))

        index_responses = self.config.vespa_client.feed_batch(vespa_docs, self.marqo_index.schema_name)

        result = []
        for resp in index_responses.responses:
            # FIXME if response is error, id is not returned
            doc_id = resp.id.split('::')[-1] if resp.id else None
            status, message = self.config.document.translate_vespa_document_response(resp.status, message=resp.message)
            result.append(MarqoAddDocumentsItem(id=doc_id, status=status, message=message))

        return result

    def _chunk_text_content(self, invalid_docs: List[MarqoAddDocumentsItem]):
        text_chunk_prefix = self.marqo_index.model.get_text_chunk_prefix(self.add_docs_params.text_chunk_prefix)
        for doc_id, field_name, tensor_field_content in (
                self.tensor_fields_container.tensor_fields_to_vectorise(FieldType.Text)):
            text_preprocessing = self.marqo_index.text_preprocessing
            # TODO handle error
            chunks = text_processor.split_text(text=tensor_field_content.field_content,
                                               split_by=text_preprocessing.split_method.value,
                                               split_length=text_preprocessing.split_length,
                                               split_overlap=text_preprocessing.split_overlap)
            tensor_field_content.chunks = chunks
            tensor_field_content.content_chunks = text_processor.prefix_text_chunks(chunks, text_chunk_prefix)

    def _batch_vectorise(self, invalid_docs: List[MarqoAddDocumentsItem]):
        field_types_to_vectorise = [FieldType.Text, FieldType.ImagePointer]

        for doc_id, field_name, tensor_field_content in (
                self.tensor_fields_container.tensor_fields_to_vectorise(*field_types_to_vectorise)):

            # TODO batch request to get more GPU utilisation, also consider how to fail fast
            # TODO handle exception, and skip doc if any field cannot be vectorised
            embeddings = s2_inference.vectorise(
                model_name=self.marqo_index.model.name,
                model_properties=self.marqo_index.model.get_properties(),
                content=tensor_field_content.content_chunks,
                device=self.add_docs_params.device,
                normalize_embeddings=self.marqo_index.normalize_embeddings,
                infer=tensor_field_content.field_type == FieldType.ImagePointer,
                model_auth=self.add_docs_params.model_auth
            )

            tensor_field_content.embeddings = embeddings

