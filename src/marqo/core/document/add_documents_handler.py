import copy
import hashlib
import os
import uuid
from abc import ABC, abstractmethod
from contextlib import ExitStack
from timeit import default_timer as timer
from typing import List, Dict, Optional, Any, Tuple, Set, Union

import torch
from PIL.Image import Image

from marqo import marqo_docs
from marqo.api import exceptions as api_errors
from marqo.config import Config
from marqo.core.constants import MARQO_DOC_ID
from marqo.core.document.models.add_docs_params import AddDocsParams
from marqo.core.document.tensor_fields_container import Chunker, Vectoriser, TensorFieldsContainer, TensorFieldContent
from marqo.core.exceptions import AddDocumentsError, DuplicateDocumentError, MarqoDocumentParsingError
from marqo.core.models import MarqoIndex
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsItem, MarqoAddDocumentsResponse
from marqo.core.models.marqo_index import FieldType
from marqo.logging import get_logger
from marqo.s2_inference import errors as s2_inference_errors
from marqo.s2_inference import s2_inference
from marqo.s2_inference.multimodal_model_load import Modality
from marqo.s2_inference.processing import image as image_processor
from marqo.s2_inference.processing import text as text_processor
from marqo.tensor_search import validation, add_docs
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.vespa.models import VespaDocument, FeedBatchResponse
from marqo.vespa.models.get_document_response import Document

logger = get_logger(__name__)


class AddDocumentsResponseCollector:
    def __init__(self):
        self.start_time = timer()
        self.responses: List[Tuple[int, MarqoAddDocumentsItem]] = []
        self.errors = False
        self.marqo_docs: Dict[str, Dict[str, Any]] = dict()
        self.marqo_doc_loc_map: Dict[str, int] = dict()

        # stores all the visited docs with _id provided by user, for dedup and retrieving existing docs
        # key is the provided _id, value is whether it's valid or not
        self.visited_doc_ids: Dict[str, bool] = dict()

    def visited(self, doc_id: str) -> bool:
        return doc_id in self.visited_doc_ids

    def valid_original_ids(self) -> Set[str]:
        return {_id for _id, valid in self.visited_doc_ids.items() if valid}

    def collect_marqo_doc(self, loc: int, marqo_doc: Dict[str, Any], original_id: Optional[str]):
        doc_id = marqo_doc[MARQO_DOC_ID]
        self.marqo_docs[doc_id] = marqo_doc
        self.marqo_doc_loc_map[doc_id] = loc
        if original_id:
            self.visited_doc_ids[original_id] = True

    def collect_error_response(self, doc_id: Optional[str], error: AddDocumentsError, loc: Optional[int] = None):
        # log errors in one place, log in warning level for each individual doc error
        # TODO it might be too verbose, but check if we need exc_info=(type(error), error, error.__traceback__)
        logger.warning(f'Encountered error when adding doc {doc_id}: {str(error)}')

        if isinstance(error, DuplicateDocumentError):
            # This is the current behaviour, docs with same id silently supersedes previous ones defined in the batch
            # TODO change the logic when we need to report duplicates as an error in the response
            return

        if doc_id and doc_id not in self.marqo_docs:
            # We mark it as visited even when there's an error. This prevents following doc with the same id from
            # being handled. doc_id not in self.marqo_docs means it's not collected yet, so the error is thrown
            # during the first validation phase
            self.visited_doc_ids[doc_id] = False

        if not loc:
            loc = self.marqo_doc_loc_map.get(doc_id)

        if doc_id in self.marqo_docs:
            self.marqo_docs.pop(doc_id, None)

        self.responses.append((loc, MarqoAddDocumentsItem(
            id=doc_id if doc_id in self.visited_doc_ids else '',
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
        Template method for adding documents to a Marqo index. This method define a generic workflow to add documents
        in batches:
        1. Traverse the docs list in reserved order to skip duplicate documents
        2. for each document, do validation first, and collect it to a dictionary
        3. tensor field information will be collected in `tensor_fields_container`
        4. Populate tensors from existing docs if `use_existing_tensors` is specified in the add_docs_params
        5. Vectorise the remaining tensor fields (including downloading, preprocessing, chunking)
        6. Convert the marqo docs to Vespa docs
        7. Persist all Vespa docs to vespa in batches
        8. Collect the response and return

        Index-type-agnostic logic are implemented in this class, and type-specific logic are extracted as abstract
        methods and implemented in add_docs_handler for individual types.
        """
        with RequestMetricsStore.for_request().time("add_documents.processing_before_vespa"):
            for loc, doc in enumerate(reversed(self.add_docs_params.docs)):
                original_id = None
                try:
                    self.validate_doc(doc)
                    # If _id is not provide, generate a ramdom one
                    original_id = doc.get(MARQO_DOC_ID)
                    marqo_doc = {MARQO_DOC_ID: original_id or str(uuid.uuid4())}

                    for field_name, field_content in doc.items():
                        if field_name == MARQO_DOC_ID:
                            continue  # we don't handle _id field
                        self.handle_field(marqo_doc, field_name, field_content)

                    self.handle_multi_modal_fields(marqo_doc)

                    self.add_docs_response_collector.collect_marqo_doc(loc, marqo_doc, original_id)
                except AddDocumentsError as err:
                    self.add_docs_response_collector.collect_error_response(original_id, err, loc)

            # retrieve existing docs for existing tensor
            if self.add_docs_params.use_existing_tensors:
                # TODO capture the telemetry data for retrieving exiting docs?
                result = self.config.vespa_client.get_batch(list(self.add_docs_response_collector.valid_original_ids()),
                                                            self.marqo_index.schema_name)
                existing_vespa_docs = [r.document for r in result.responses if r.status == 200]
                self.handle_existing_tensors(existing_vespa_docs)

            # vectorise tensor fields
            self.vectorise_tensor_fields()

        # FIXME this step is not timed in the original implementation
        vespa_docs = self.convert_to_vespa_docs()

        self.pre_persist_to_vespa()

        # persist to vespa if there are still valid docs
        with RequestMetricsStore.for_request().time("add_documents.vespa._bulk"):
            response = self.config.vespa_client.feed_batch(vespa_docs, self.marqo_index.schema_name)

        with RequestMetricsStore.for_request().time("add_documents.postprocess"):
            self.handle_vespa_response(response)
            return self.add_docs_response_collector.to_add_doc_responses(self.marqo_index.name)

    @abstractmethod
    def _create_tensor_fields_container(self) -> TensorFieldsContainer:
        """
        This method generates a tensor fields container using information in marqo_index and add_docs_params.
        The information includes the tensor fields, mappings, etc.
        """
        pass

    @abstractmethod
    def handle_field(self, marqo_doc, field_name, field_content) -> None:
        """
        This method handles each individual field in a marqo doc, validates it, collect tensor info into
        `tensor_fields_container`, and change the field content if necessary (e.g. custom vector fields)
        """
        pass

    @abstractmethod
    def handle_multi_modal_fields(self, marqo_doc: Dict[str, Any]) -> None:
        """
        This method collect the information for multimodal combo fields in a Marqo doc.
        """
        pass

    @abstractmethod
    def handle_existing_tensors(self, existing_vespa_docs: List[Document]) -> None:
        """
        This method populates embeddings from existing documents. We could save some resources and time
        by skipping vectorisation of existing tensor fields with the same content.
        """
        pass

    @abstractmethod
    def to_vespa_doc(self, marqo_doc: Dict[str, Any]) -> VespaDocument:
        """
        Convert a marqo doc into a VespaDocument.
        """
        pass

    def pre_persist_to_vespa(self) -> None:
        """
        A hook method to do extra handling before we persist docs to Vespa. By default, it does nothing
        """
        pass

    def convert_to_vespa_docs(self) -> List[VespaDocument]:
        vespa_docs = []
        for doc_id, doc in self.add_docs_response_collector.marqo_docs.copy().items():
            try:
                vespa_docs.append(self.to_vespa_doc(doc))
            except MarqoDocumentParsingError as e:
                self.add_docs_response_collector.collect_error_response(doc_id, AddDocumentsError(e.message))

        return list(reversed(vespa_docs))

    def handle_vespa_response(self, response: FeedBatchResponse):
        for resp in response.responses:
            # FIXME doc_id is not url encoded
            doc_id = resp.id.split('::')[-1] if resp.id else None
            status, message = self.config.document.translate_vespa_document_response(resp.status, message=resp.message)
            if status != 200:
                self.add_docs_response_collector.collect_error_response(doc_id, AddDocumentsError(
                    error_message=message, status_code=status, error_code='vespa_error'  # breaking?
                ))
            else:
                self.add_docs_response_collector.collect_successful_response(doc_id)

    def validate_doc(self, doc) -> None:
        try:
            validation.validate_doc(doc)

            if MARQO_DOC_ID in doc:
                # validate _id field
                doc_id = doc[MARQO_DOC_ID]
                validation.validate_id(doc_id)
                if self.add_docs_response_collector.visited(doc_id):
                    raise DuplicateDocumentError(f"Document will be ignored since doc with the same id"
                                                 f" `{doc_id}` supersedes this one")

        except (api_errors.InvalidArgError, api_errors.DocTooLargeError, api_errors.InvalidDocumentIdError) as err:
            raise AddDocumentsError(err.message, error_code=err.code, status_code=err.status_code) from err

    # The following code are about handling all tensor fields
    # TODO see if we should move these code to some other classes. They are kept here due to the dependency on
    #   both the marqo_index and add_docs_params

    MODALITY_FIELD_TYPE_MAP = {
        Modality.TEXT: FieldType.Text,
        Modality.IMAGE: FieldType.ImagePointer,
        Modality.VIDEO: FieldType.VideoPointer,
        Modality.AUDIO: FieldType.AudioPointer,
    }

    def vectorise_tensor_fields(self) -> None:
        """
        Download, preprocess, chunk and vectorise collected tensor fields.
        Three different batching strategies can be chosen to do tradeoff between resource usage and performance.
        - Batching by field: Chunk and vectorise field by field (Default?)
        - Batching by doc: Chunk and vectorise fields of a doc by field type (text, image, audio, video, etc.)
        - Batching by add-doc batch: Chunk and vectorise all fields of a batch of docs by type
        """
        # TODO add a parameter to choose batching strategy? Do a performance test to decide the default approach
        # self.vectorise_tensor_fields_per_field()
        self.vectorise_tensor_fields_in_batch_per_doc()
        # self.vectorise_tensor_fields_in_batch_per_add_doc_batch()

    def vectorise_tensor_fields_per_field(self) -> None:
        with ExitStack() as exit_stack:
            media_repo = self._download_media_contents(exit_stack)
            chunkers = self._field_type_chunker_map(media_repo)
            vectorisers = {field_type: self.single_vectoriser(modality)
                           for modality, field_type in self.MODALITY_FIELD_TYPE_MAP.items()}

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
            media_repo = self._download_media_contents(exit_stack)
            chunkers = self._field_type_chunker_map(media_repo)

            doc_chunks_map: Dict[str, Dict[FieldType, List[str]]] = dict()
            doc_field_map: Dict[str, List[TensorFieldContent]] = dict()

            for doc_id, field_name, tensor_field_content in (
                    self.tensor_fields_container.tensor_fields_to_vectorise(*chunkers.keys())):
                try:
                    tensor_field_content.chunk(chunkers)
                    doc_chunks_map.setdefault(doc_id, {}).setdefault(
                        tensor_field_content.field_type, []).extend(tensor_field_content.content_chunks)
                    doc_field_map.setdefault(doc_id, []).append(tensor_field_content)

                except AddDocumentsError as err:
                    self.add_docs_response_collector.collect_error_response(doc_id, err)
                    self.tensor_fields_container.remove_doc(doc_id)
                    if doc_id in doc_chunks_map:
                        del doc_chunks_map[doc_id]

            # TODO check if we should capture total vectorise time
            for doc_id, chunks_to_vectorise in doc_chunks_map.items():
                try:
                    vectorisers = {field_type: self.batch_vectoriser(chunks_to_vectorise[field_type], modality)
                                   for modality, field_type in self.MODALITY_FIELD_TYPE_MAP.items()
                                   if field_type in chunks_to_vectorise}

                    for tensor_field_content in doc_field_map[doc_id]:
                        tensor_field_content.vectorise(vectorisers)

                except AddDocumentsError as err:
                    self.add_docs_response_collector.collect_error_response(doc_id, err)
                    self.tensor_fields_container.remove_doc(doc_id)

    def vectorise_tensor_fields_in_batch_per_add_doc_batch(self) -> None:
        with ExitStack() as exit_stack:
            media_repo = self._download_media_contents(exit_stack)
            chunkers = self._field_type_chunker_map(media_repo)
            chunks_map = {field_type: [] for field_type in chunkers.keys()}

            for doc_id, field_name, tensor_field_content in (
                    self.tensor_fields_container.tensor_fields_to_vectorise(*chunkers.keys())):
                try:
                    tensor_field_content.chunk(chunkers)
                    chunks_to_vectorise = tensor_field_content.content_chunks
                    field_type = tensor_field_content.field_type
                    chunks_map[field_type].extend(chunks_to_vectorise)
                except AddDocumentsError as err:
                    self.add_docs_response_collector.collect_error_response(doc_id, err)
                    self.tensor_fields_container.remove_doc(doc_id)

            try:
                vectorisers = {field_type: self.batch_vectoriser(chunks_map[field_type], modality)
                               for modality, field_type in self.MODALITY_FIELD_TYPE_MAP.items()}
            except AddDocumentsError as err:
                # TODO check if it is too verbose to log out traceback
                logger.error('Encountered problem when vectorising batch of documents. Reason: %s', err, exc_info=True)
                # TODO raise a core exception
                raise api_errors.BadRequestError(
                    message=f'Encountered problem when vectorising batch of documents. Reason: {str(err)}'
                )

            for doc_id, field_name, tensor_field_content in (
                    self.tensor_fields_container.tensor_fields_to_vectorise(*chunkers.keys())):
                tensor_field_content.vectorise(vectorisers)

    def _field_type_chunker_map(self, media_repo):
        chunkers: Dict[FieldType, Chunker] = {
            FieldType.Text: self.text_chunker(),
            FieldType.ImagePointer: self.image_chunker(media_repo),
            FieldType.AudioPointer: self.video_audio_chunker(media_repo),
            FieldType.VideoPointer: self.video_audio_chunker(media_repo),
        }
        return chunkers

    def _download_media_contents(self, exit_stack):
        url_doc_id_map = dict()
        doc_media_fields = dict()
        media_field_types_mapping = dict()

        media_field_types = [FieldType.ImagePointer, FieldType.AudioPointer, FieldType.VideoPointer]

        for doc_id, field_name, tensor_field_content in (
                self.tensor_fields_container.tensor_fields_to_vectorise(*media_field_types)):
            url = tensor_field_content.field_content
            url_doc_id_map.setdefault(url, set()).add(doc_id)
            doc_media_fields.setdefault(doc_id, dict())[field_name] = url
            media_field_types_mapping[field_name] = tensor_field_content.field_type

        if not doc_media_fields:
            return dict()

        with RequestMetricsStore.for_request().time("image_download.full_time"):
            image_repo = exit_stack.enter_context(
                # TODO refactor this to only pass in necessary parameters
                add_docs.download_and_preprocess_content(
                    docs=list(doc_media_fields.values()),
                    thread_count=self._determine_thread_count(self.marqo_index, self.add_docs_params),
                    tensor_fields=list(media_field_types_mapping.keys()),
                    image_download_headers=self.add_docs_params.image_download_headers,
                    model_name=self.marqo_index.model.name,
                    normalize_embeddings=self.marqo_index.normalize_embeddings,
                    media_field_types_mapping=media_field_types_mapping,
                    model_properties=self.marqo_index.model.get_properties(),
                    device=self.add_docs_params.device,
                    model_auth=self.add_docs_params.model_auth,
                    patch_method_exists=self.marqo_index.image_preprocessing.patch_method is not None,
                    marqo_index_type=self.marqo_index.type,
                    marqo_index_model=self.marqo_index.model,
                    audio_preprocessing=self.marqo_index.audio_preprocessing,
                    video_preprocessing=self.marqo_index.video_preprocessing,
                )
            )

        for url, data in image_repo.items():
            if isinstance(data, Exception):
                for doc_id in url_doc_id_map[url]:
                    self.add_docs_response_collector.collect_error_response(doc_id, AddDocumentsError(
                        error_message=f"Could not process the media file found at `{url}`. Reason: {str(data)}"
                    ))
                    self.tensor_fields_container.remove_doc(doc_id)

        return image_repo

    def _determine_thread_count(self, marqo_index: MarqoIndex, add_docs_params: AddDocsParams):
        # TODO this logic is copied from tensor search. Can be simplified and moved to AddDocsParams?
        model_properties = marqo_index.model.get_properties()
        is_languagebind_model = model_properties.get('type') == 'languagebind'

        default_image_thread_count = 20
        default_media_thread_count = 5

        # Check if media_download_thread_count is set in params
        if (add_docs_params.media_download_thread_count is not None and
                add_docs_params.media_download_thread_count != default_media_thread_count):
            return add_docs_params.media_download_thread_count

        env_media_thread_count = os.environ.get(EnvVars.MARQO_MEDIA_DOWNLOAD_THREAD_COUNT_PER_REQUEST)
        if env_media_thread_count is not None and int(env_media_thread_count) != default_media_thread_count:
            return int(env_media_thread_count)

        # If it's a LanguageBind model and no explicit setting, use 5
        if is_languagebind_model:
            return 5

        # Check if image_download_thread_count is explicitly set in params
        if (add_docs_params.image_download_thread_count is not None and
                add_docs_params.image_download_thread_count != default_image_thread_count):
            return add_docs_params.image_download_thread_count

        # Check if environment variable is explicitly set
        env_image_thread_count = os.environ.get(EnvVars.MARQO_IMAGE_DOWNLOAD_THREAD_COUNT_PER_REQUEST)
        if env_image_thread_count is not None and int(env_image_thread_count) != default_image_thread_count:
            return int(env_image_thread_count)

        # Default case
        return default_image_thread_count

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

    def image_chunker(self, media_repo) -> Chunker:
        image_method = self.marqo_index.image_preprocessing.patch_method

        def chunk(field_content: str, single_chunk: bool = False):
            url = field_content
            image_data = media_repo[url]
            if single_chunk or image_method is None:
                return [url], [image_data]

            try:
                content_chunks, text_chunks = image_processor.chunk_image(
                    image_data, device=self.add_docs_params.device, method=image_method.value)
                return text_chunks, content_chunks
            except s2_inference_errors.S2InferenceError as e:
                raise AddDocumentsError(e.message) from e

        return chunk

    def video_audio_chunker(self, media_repo) -> Chunker:
        def chunk_id(media_chunk: dict) -> str:
            chunk_start = media_chunk['start_time']
            chunk_end = media_chunk['end_time']
            return f"{[chunk_start, chunk_end]}"

        def chunk(field_content: str, single_chunk: bool = False):
            url = field_content
            media_chunks = media_repo[url]

            # single_chunk does not apply to video and audio fields. Instead, it needs to calculate the
            # embedding for each chunk and average them
            if single_chunk:
                raise RuntimeError("Video and Audio chunker does not support single_chunk")

            text_chunks = [chunk_id(media_chunk) for media_chunk in media_chunks]
            content_chunks = [media_chunk['tensor'] for media_chunk in media_chunks]

            return text_chunks, content_chunks

        return chunk

    def single_vectoriser(self, modality: Modality) -> Vectoriser:
        def vectorise(content_chunks: Union[List[str], List[Image]]) -> List[List[float]]:
            with RequestMetricsStore.for_request().time(f"add_documents.create_vectors"):
                if modality in [Modality.AUDIO, Modality.VIDEO]:
                    # audio and video fields has to be vectorised chunk by chunk due to a limitation of languagebind
                    return [vector for content_chunk in content_chunks for vector in
                            self._s2inference_vectorise([content_chunk], modality)]
                return self._s2inference_vectorise(content_chunks, modality)

        return vectorise

    def batch_vectoriser(self, chunks_to_vectorise: Union[List[str], List[Image]], modality: Modality) -> Vectoriser:
        def dict_key(chunk: Union[str, Image, torch.Tensor, Dict[str, torch.Tensor]]):
            if isinstance(chunk, Image):
                chunk = chunk.convert('RGB')
                pixel_bytes = chunk.tobytes()
                # Use md5 hash for faster hashing.
                return hashlib.md5(pixel_bytes).hexdigest()
            elif isinstance(chunk, dict):
                # Generate a sorted key-value pairs to ensure consistency.
                return frozenset((k, dict_key(v)) for k, v in chunk.items())
            elif isinstance(chunk, torch.Tensor):
                # Convert to a tuple to be hashable  # TODO find a more memory efficient way, maybe hashlib.md5?
                return tuple(chunk.flatten().tolist())
            else:
                return chunk

        embedding_cache = dict()
        if chunks_to_vectorise:
            # TODO this might be a breaking change since the create_vectors metrics is not consistent
            #   for individual tensor fields and subfields of multimodal fields
            with RequestMetricsStore.for_request().time(f"add_documents.create_vectors"):
                if modality in [Modality.AUDIO, Modality.VIDEO]:
                    # audio and video fields has to be vectorised chunk by chunk due to a limitation of languagebind
                    embeddings = [vector for content_chunk in chunks_to_vectorise for vector in
                                  self._s2inference_vectorise([content_chunk], modality)]
                else:
                    embeddings = self._s2inference_vectorise(chunks_to_vectorise, modality)
                embedding_cache = {dict_key(chunk): embeddings[i] for i, chunk in enumerate(chunks_to_vectorise)}

        def vectorise(content_chunks: Union[List[str], List[Image]]) -> List[List[float]]:
            return [embedding_cache[dict_key(chunk)] for chunk in content_chunks]

        return vectorise

    def _s2inference_vectorise(self, content_chunks: Union[List[str], List[Image]], modality: Modality):
        try:
            return s2_inference.vectorise(
                model_name=self.marqo_index.model.name,
                model_properties=self.marqo_index.model.get_properties(),
                content=content_chunks,
                device=self.add_docs_params.device,
                normalize_embeddings=self.marqo_index.normalize_embeddings,
                model_auth=self.add_docs_params.model_auth,
                modality=modality
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
            raise AddDocumentsError(e.message) from e

