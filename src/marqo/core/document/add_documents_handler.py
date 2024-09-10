import copy
import json
import uuid
from abc import ABC, abstractmethod
from contextlib import ExitStack
from http import HTTPStatus
from timeit import default_timer as timer
from typing import List, Dict, Optional, Any, Tuple, Set, Union

from PIL.Image import Image

from marqo import marqo_docs
from marqo.config import Config
from marqo.core.constants import MARQO_DOC_ID
from marqo.core.document.models.add_docs_params import AddDocsParams
from marqo.core.document.tensor_fields_container import Chunker, Vectoriser, TensorFieldsContainer, TensorFieldContent
from marqo.core.exceptions import AddDocumentsError, DuplicateDocumentError, MarqoDocumentParsingError
from marqo.core.models import MarqoIndex
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsItem, MarqoAddDocumentsResponse
from marqo.core.models.marqo_index import FieldType
from marqo.tensor_search import utils, enums, validation, add_docs
from marqo.vespa.models import VespaDocument
from marqo.vespa.models.get_document_response import Document
from marqo.api import exceptions as api_errors

from marqo.s2_inference.processing import image as image_processor
from marqo.s2_inference.processing import text as text_processor
from marqo.s2_inference import s2_inference
from marqo.s2_inference import errors as s2_inference_errors


ORIGINAL_ID='_original_id'


class AddDocumentsResponseCollector:
    def __init__(self):
        self.start_time = timer()
        # TODO we ignore the location for now, and will add it if needed in the future
        self.responses: List[Tuple[int, MarqoAddDocumentsItem]] = []
        self.errors = False
        self.marqo_docs: Dict[str, Dict[str, Any]] = dict()
        self.marqo_doc_loc_map: Dict[str, int] = dict()
        self.visited_doc_ids: Set[str] = set()

    def visited(self, doc_id: str) -> bool:
        return doc_id in self.visited_doc_ids

    def collect_marqo_doc(self, loc: int, marqo_doc: Dict[str, Any]):
        doc_id = marqo_doc[MARQO_DOC_ID]
        self.marqo_docs[doc_id] = marqo_doc
        self.marqo_doc_loc_map[doc_id] = loc
        if marqo_doc[ORIGINAL_ID] is not None:
            self.visited_doc_ids.add(marqo_doc[ORIGINAL_ID])

    def collect_error_response(self, doc_id: Optional[str], error: AddDocumentsError, loc: Optional[int] = None):
        if isinstance(error, DuplicateDocumentError):
            # This is the current logic, docs with same id supersedes previous ones defined in the batch
            # TODO change the logic when we need to report duplicates as error in the response
            return

        if not loc and doc_id and doc_id in self.marqo_doc_loc_map:
            loc = self.marqo_doc_loc_map[doc_id]

        if doc_id in self.marqo_docs:
            doc_id = self.marqo_docs.pop(doc_id)[ORIGINAL_ID]

        # TODO log errors in one place

        # Even if the last document is invalid, we should not use previous ones?
        if doc_id:
            self.visited_doc_ids.add(doc_id)

        self.responses.append((loc, MarqoAddDocumentsItem(
            id=doc_id if doc_id is not None else '',
            error=error.error_message,
            message=error.error_message,
            status=error.status_code,
            code=error.error_code
        )))

        self.errors = True

    def collect_successful_response(self, doc_id: Optional[str]):
        loc = self.marqo_doc_loc_map.get(doc_id, None)

        self.responses.append((loc, MarqoAddDocumentsItem(
            id=doc_id if doc_id is not None else '',
            status=200,
        )))

    def to_add_doc_responses(self, index_name: str) -> MarqoAddDocumentsResponse:
        processing_time = (timer() - self.start_time) * 1000
        # since we reversed the doc list to skip duplicate docs, we now need to reverse the response
        sorted_responses = [response for _, response in sorted(self.responses, key=lambda r: r[0] or 0, reverse=True)]
        return MarqoAddDocumentsResponse(errors=self.errors, index_name=index_name, items=sorted_responses,
                                         processingTimeMs=processing_time)


class AddDocumentsHandler(ABC):

    def __init__(self, marqo_index: MarqoIndex, config: Config, add_docs_params: AddDocsParams):
        self.marqo_index = marqo_index
        self.add_docs_params = add_docs_params
        self.config = config
        self.add_docs_response_collector = AddDocumentsResponseCollector()
        self.tensor_fields_container = self._create_tensor_fields_container()

    def add_documents(self):
        """
        Template method for adding documents to Marqo index
        """
        for loc, original_doc in enumerate(reversed(self.add_docs_params.docs)):
            doc = copy.deepcopy(original_doc)
            original_id = None
            try:
                self.validate_doc(doc)

                original_id = self.validate_and_pop_doc_id(doc)
                doc_id = original_id or str(uuid.uuid4())
                marqo_doc = {ORIGINAL_ID: original_id, MARQO_DOC_ID: doc_id}  # keep this info for error report

                for field_name, field_content in doc.items():
                    self.handle_field(marqo_doc, field_name, field_content)

                self.handle_multi_modal_fields(marqo_doc)

                self.add_docs_response_collector.collect_marqo_doc(loc, marqo_doc)
            except AddDocumentsError as err:
                self.add_docs_response_collector.collect_error_response(original_id, err, loc)

        # retrieve existing docs for existing tensor
        if self.add_docs_params.use_existing_tensors:
            result = self.config.vespa_client.get_batch(list(self.add_docs_response_collector.visited_doc_ids),
                                                        self.marqo_index.schema_name)
            existing_vespa_docs = {r.id: r.document for r in result.responses if r.status == 200}
            self.handle_existing_tensors(existing_vespa_docs)

        # vectorise tensor fields
        self.vectorise_tensor_fields()

        # persist to vespa if there are still valid docs
        self.persist_to_vespa()

        return self.add_docs_response_collector.to_add_doc_responses(self.marqo_index.name)

    @abstractmethod
    def _create_tensor_fields_container(self) -> TensorFieldsContainer:
        pass

    @abstractmethod
    def handle_field(self, marqo_doc, field_name, field_content):
        pass

    @abstractmethod
    def handle_multi_modal_fields(self, marqo_doc: Dict[str, Any]):
        pass

    @abstractmethod
    def handle_existing_tensors(self, existing_vespa_docs: Dict[str, Document]):
        pass

    def vectorise_tensor_fields(self) -> None:
        self.vectorise_tensor_fields_in_batch_per_doc()

    def vectorise_tensor_fields_per_field(self) -> None:
        with ExitStack() as exit_stack:
            chunkers: Dict[FieldType, Chunker] = {
                FieldType.Text: self.text_chunker(),
                FieldType.ImagePointer: self.image_chunker(exit_stack)
            }
            vectorisers: Dict[FieldType, Vectoriser] = {
                FieldType.Text: self.single_vectoriser(infer=False),
                FieldType.ImagePointer: self.single_vectoriser(infer=True)
            }

            for doc_id, field_name, tensor_field_content in (
                    self.tensor_fields_container.tensor_fields_to_vectorise(*chunkers.keys())):
                try:
                    tensor_field_content.chunk(chunkers)
                    tensor_field_content.vectorise(vectorisers)
                except AddDocumentsError as err:
                    self.add_docs_response_collector.collect_error_response(doc_id, err)
                    self.tensor_fields_container.remove_doc(doc_id)

    def vectorise_tensor_fields_in_batch_per_doc(self) -> None:
        with ExitStack() as exit_stack:
            chunkers: Dict[FieldType, Chunker] = {
                FieldType.Text: self.text_chunker(),
                FieldType.ImagePointer: self.image_chunker(exit_stack)
            }

            doc_chunks_map: Dict[str, Dict[FieldType, List[str]]] = dict()
            doc_field_map: Dict[str, List[TensorFieldContent]] = dict()

            for doc_id, field_name, tensor_field_content in (
                    self.tensor_fields_container.tensor_fields_to_vectorise(*chunkers.keys())):
                try:
                    chunks_to_vectorise = tensor_field_content.chunk(chunkers)
                    field_type = tensor_field_content.field_type
                    if doc_id not in doc_chunks_map:
                        doc_chunks_map[doc_id] = {
                            FieldType.Text: [],
                            FieldType.ImagePointer: []
                        }
                    doc_chunks_map[doc_id][field_type].extend(chunks_to_vectorise)

                    if doc_id not in doc_field_map:
                        doc_field_map[doc_id] = []
                    doc_field_map[doc_id].append(tensor_field_content)

                except AddDocumentsError as err:
                    self.add_docs_response_collector.collect_error_response(doc_id, err)
                    self.tensor_fields_container.remove_doc(doc_id)
                    if doc_id in doc_chunks_map:
                        del doc_chunks_map[doc_id]

            for doc_id, chunks_to_vectorise in doc_chunks_map.items():
                try:
                    vectorisers: Dict[FieldType, Vectoriser] = {
                        FieldType.Text: self.batch_vectoriser(chunks_to_vectorise[FieldType.Text], infer=False),
                        FieldType.ImagePointer: self.batch_vectoriser(chunks_to_vectorise[FieldType.ImagePointer], infer=True)
                    }

                    for tensor_field_content in doc_field_map[doc_id]:
                        tensor_field_content.vectorise(vectorisers)

                except AddDocumentsError as err:
                    self.add_docs_response_collector.collect_error_response(doc_id, err)
                    self.tensor_fields_container.remove_doc(doc_id)

    def vectorise_tensor_fields_in_batch_per_add_doc_batch(self) -> None:
        with ExitStack() as exit_stack:
            chunkers: Dict[FieldType, Chunker] = {
                FieldType.Text: self.text_chunker(),
                FieldType.ImagePointer: self.image_chunker(exit_stack)
            }

            chunks_map: Dict[FieldType, List[str]] = {
                FieldType.Text: [],
                FieldType.ImagePointer: []
            }

            for doc_id, field_name, tensor_field_content in (
                    self.tensor_fields_container.tensor_fields_to_vectorise(*chunkers.keys())):
                try:
                    chunks_to_vectorise = tensor_field_content.chunk(chunkers)
                    field_type = tensor_field_content.field_type
                    chunks_map[field_type].extend(chunks_to_vectorise)
                except AddDocumentsError as err:
                    self.add_docs_response_collector.collect_error_response(doc_id, err)
                    self.tensor_fields_container.remove_doc(doc_id)

            try:
                vectorisers: Dict[FieldType, Vectoriser] = {
                    FieldType.Text: self.batch_vectoriser(chunks_to_vectorise[FieldType.Text], infer=False),
                    FieldType.ImagePointer: self.batch_vectoriser(chunks_to_vectorise[FieldType.ImagePointer], infer=True)
                }
            except AddDocumentsError as err:
                # TODO we need to fail the batch
                return

            for doc_id, field_name, tensor_field_content in (
                    self.tensor_fields_container.tensor_fields_to_vectorise(*chunkers.keys())):
                tensor_field_content.vectorise(vectorisers)

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

    def single_vectoriser(self, infer: bool) -> Vectoriser:
        def vectorise(content_chunks: Union[List[str], List[Image]]) -> List[List[float]]:
            return self._s2inference_vectorise(content_chunks, infer)

        return vectorise

    def batch_vectoriser(self, chunks_to_vectorise: Union[List[str], List[Image]], infer: bool) -> Vectoriser:
        def dict_key(chunk: Union[str, Image]):
            if isinstance(chunk, str):
                return chunk
            if isinstance(chunk, Image):
                return hash((chunk.format, chunk.size))
            raise AddDocumentsError("Content chunk is not str or Image")

        embedding_cache = dict()
        if chunks_to_vectorise:
            embeddings = self._s2inference_vectorise(chunks_to_vectorise, infer)
            embedding_cache = {dict_key(chunk): embeddings[i] for i, chunk in enumerate(chunks_to_vectorise)}

        def vectorise(content_chunks: Union[List[str], List[Image]]) -> List[List[float]]:
            return [embedding_cache[dict_key(chunk)] for chunk in content_chunks]

        return vectorise

    def _s2inference_vectorise(self, content_chunks: Union[List[str], List[Image]], infer: bool):
        try:
            return s2_inference.vectorise(
                model_name=self.marqo_index.model.name,
                model_properties=self.marqo_index.model.get_properties(),
                content=content_chunks,
                device=self.add_docs_params.device,
                normalize_embeddings=self.marqo_index.normalize_embeddings,
                model_auth=self.add_docs_params.model_auth,
                # TODO how is this param used? Is it different for different modals?
                infer=infer
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

    def persist_to_vespa(self) -> None:
        vespa_docs = []
        for doc_id, doc in self.add_docs_response_collector.marqo_docs.copy().items():
            try:
                vespa_docs.append(self.to_vespa_doc(doc))
            except MarqoDocumentParsingError as e:
                self.add_docs_response_collector.collect_error_response(doc_id, AddDocumentsError(e.message))

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

    @abstractmethod
    def to_vespa_doc(self, marqo_doc: Dict[str, Any]) -> VespaDocument:
        pass

    def validate_doc(self, doc) -> None:
        try:
            validation.validate_doc(doc)
        except (api_errors.InvalidArgError, api_errors.DocTooLargeError) as err:
            raise AddDocumentsError(err.message, error_code=err.code, status_code=err.status_code) from err

    def validate_and_pop_doc_id(self, doc) -> Optional[str]:
        if MARQO_DOC_ID not in doc:
            return None

        doc_id = doc.pop(MARQO_DOC_ID)
        try:
            validation.validate_id(doc_id)
        except api_errors.InvalidDocumentIdError as err:
            raise AddDocumentsError(err.message, error_code=err.code, status_code=err.status_code) from err

        if self.add_docs_response_collector.visited(doc_id):
            raise DuplicateDocumentError(f"Document will be ignored since doc with the same id"
                                         f" `{doc_id}` supersedes this one")

        return doc_id

