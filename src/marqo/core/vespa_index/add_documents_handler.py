import uuid
from abc import ABC, abstractmethod
from contextlib import ExitStack
from timeit import default_timer as timer
from typing import List, Dict, Optional, Any, Tuple, Set

from marqo.api import exceptions as api_errors
from marqo.core.constants import MARQO_DOC_ID, MARQO_CUSTOM_VECTOR_NORMALIZATION_MINIMUM_VERSION
from marqo.core.models.add_docs_params import AddDocsParams, BatchVectorisationMode
from marqo.core.inference.tensor_fields_container import Chunker, TensorFieldsContainer, TensorFieldContent, \
    TextChunker, ImageChunker, AudioVideoChunker, ModelConfig, Vectoriser
from marqo.core.exceptions import AddDocumentsError, DuplicateDocumentError, MarqoDocumentParsingError, InternalError, \
    UnsupportedFeatureError
from marqo.core.models import MarqoIndex
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsItem, MarqoAddDocumentsResponse
from marqo.core.models.marqo_index import FieldType
from marqo.logging import get_logger
from marqo.tensor_search import validation, add_docs
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.vespa.models import VespaDocument, FeedBatchResponse
from marqo.vespa.models.get_document_response import Document
from marqo.vespa.vespa_client import VespaClient

logger = get_logger(__name__)


class AddDocumentsResponseCollector:
    """
    During the processing of add document batches, errors could be raised in every step. This class collects the failed
    and successful result of each individual documents along the way, and generates the final response containing this
    information.
    """
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
    """
    This class contains all the generic logic of batch adding document for all type of indexes.
    It has a template method of `add_documents` that implements the main workflow of batch adding documents and allow
    its subclass to fill in the different logic for handling individual fields, existing tensors and converting Marqo
    docs to Vespa docs, etc.
    """

    def __init__(self, marqo_index: MarqoIndex, add_docs_params: AddDocsParams, vespa_client: VespaClient):
        self.marqo_index = marqo_index
        self.add_docs_params = add_docs_params
        self.vespa_client = vespa_client
        # only normalise custom vector in new indexes to keep the backward compatibility
        self.should_normalise_custom_vector = (marqo_index.normalize_embeddings and marqo_index.parsed_marqo_version()
                                               >= MARQO_CUSTOM_VECTOR_NORMALIZATION_MINIMUM_VERSION)
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
                    self._validate_doc(doc)
                    # If _id is not provide, generate a ramdom one
                    original_id = doc.get(MARQO_DOC_ID)
                    marqo_doc = {MARQO_DOC_ID: original_id or str(uuid.uuid4())}

                    for field_name, field_content in doc.items():
                        if field_name == MARQO_DOC_ID:
                            continue  # we don't handle _id field
                        self._handle_field(marqo_doc, field_name, field_content)

                    self._handle_multi_modal_fields(marqo_doc)

                    self.add_docs_response_collector.collect_marqo_doc(loc, marqo_doc, original_id)
                except AddDocumentsError as err:
                    self.add_docs_response_collector.collect_error_response(original_id, err, loc)

            # retrieve existing docs for existing tensor
            if self.add_docs_params.use_existing_tensors:
                # TODO capture the telemetry data for retrieving exiting docs?
                result = self.vespa_client.get_batch(list(self.add_docs_response_collector.valid_original_ids()),
                                                            self.marqo_index.schema_name)
                existing_vespa_docs = [r.document for r in result.responses if r.status == 200]
                self._populate_existing_tensors(existing_vespa_docs)

            # vectorise tensor fields
            self._vectorise_tensor_fields()

        # FIXME this step is not timed in the original implementation
        vespa_docs = self._convert_to_vespa_docs()

        self._pre_persist_to_vespa()

        # persist to vespa if there are still valid docs
        with RequestMetricsStore.for_request().time("add_documents.vespa._bulk"):
            response = self.vespa_client.feed_batch(vespa_docs, self.marqo_index.schema_name)

        with RequestMetricsStore.for_request().time("add_documents.postprocess"):
            self._handle_vespa_response(response)
            return self.add_docs_response_collector.to_add_doc_responses(self.marqo_index.name)

    @abstractmethod
    def _create_tensor_fields_container(self) -> TensorFieldsContainer:
        """
        This method generates a tensor fields container using information in marqo_index and add_docs_params.
        The information includes the tensor fields, mappings, etc.
        """
        pass

    @abstractmethod
    def _handle_field(self, marqo_doc, field_name, field_content) -> None:
        """
        This method handles each individual field in a marqo doc, validates it, collect tensor info into
        `tensor_fields_container`, and change the field content if necessary (e.g. custom vector fields)
        """
        pass

    @abstractmethod
    def _handle_multi_modal_fields(self, marqo_doc: Dict[str, Any]) -> None:
        """
        This method collect the information for multimodal combo fields in a Marqo doc.
        """
        pass

    @abstractmethod
    def _populate_existing_tensors(self, existing_vespa_docs: List[Document]) -> None:
        """
        This method populates embeddings from existing documents. We could save some resources and time
        by skipping vectorisation of existing tensor fields with the same content.
        """
        pass

    @abstractmethod
    def _to_vespa_doc(self, marqo_doc: Dict[str, Any]) -> VespaDocument:
        """
        Convert a marqo doc into a VespaDocument.
        """
        pass

    def _pre_persist_to_vespa(self) -> None:
        """
        A hook method to do extra handling before we persist docs to Vespa. By default, it does nothing
        """
        pass

    def _convert_to_vespa_docs(self) -> List[VespaDocument]:
        vespa_docs = []
        for doc_id, doc in self.add_docs_response_collector.marqo_docs.copy().items():
            try:
                vespa_docs.append(self._to_vespa_doc(doc))
            except MarqoDocumentParsingError as e:
                self.add_docs_response_collector.collect_error_response(doc_id, AddDocumentsError(e.message))

        return list(reversed(vespa_docs))

    def _handle_vespa_response(self, response: FeedBatchResponse):
        for resp in response.responses:
            # FIXME doc_id is not url encoded
            doc_id = resp.id.split('::')[-1] if resp.id else None
            status, message = self.vespa_client.translate_vespa_document_response(resp.status, message=resp.message)
            if status != 200:
                self.add_docs_response_collector.collect_error_response(doc_id, AddDocumentsError(
                    error_message=message, status_code=status, error_code='vespa_error'  # breaking?
                ))
            else:
                self.add_docs_response_collector.collect_successful_response(doc_id)

    def _validate_doc(self, doc) -> None:
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

    def _vectorise_tensor_fields(self) -> None:
        """
        Download, preprocess, chunk and vectorise collected tensor fields.
        Three different batching strategies can be chosen to do tradeoff between resource usage and performance.
        - Batching by field: Chunk and vectorise field by field (Default?)
        - Batching by doc: Chunk and vectorise fields of a doc by field type (text, image, audio, video, etc.)
        - Batching by add-doc batch: Chunk and vectorise all fields of a batch of docs by type
        """
        model_config = ModelConfig(
            model_name=self.marqo_index.model.name,
            model_properties=self.marqo_index.model.get_properties(),
            model_auth=self.add_docs_params.model_auth,
            device=self.add_docs_params.device,
            normalize_embeddings=self.marqo_index.normalize_embeddings
        )
        batch_mode = self.add_docs_params.batch_vectorisation_mode

        if batch_mode == BatchVectorisationMode.PER_FIELD:
            self._vectorise_tensor_fields_per_field(model_config)
        elif batch_mode == BatchVectorisationMode.PER_DOCUMENT:
            self._vectorise_tensor_fields_in_batch_per_doc(model_config)
        elif batch_mode == BatchVectorisationMode.PER_BATCH:
            self._vectorise_tensor_fields_in_batch_per_add_doc_batch(model_config)
        else:
            raise UnsupportedFeatureError(
                message=f'Unsupported batch vectorisation mode: {str(batch_mode)}'
            )

    def _vectorise_tensor_fields_per_field(self, model_config: ModelConfig) -> None:
        with ExitStack() as exit_stack:
            media_repo = self._download_media_contents(exit_stack)
            chunkers = self._field_type_chunker_map(media_repo)
            vectorisers = Vectoriser.single_vectorisers_by_modality(model_config)

            for doc_id, field_name, tensor_field_content in (
                    self.tensor_fields_container.tensor_fields_to_vectorise(*chunkers.keys())):
                try:
                    tensor_field_content.chunk(chunkers)
                    tensor_field_content.vectorise(vectorisers)
                except AddDocumentsError as err:
                    self.add_docs_response_collector.collect_error_response(doc_id, err)
                    self.tensor_fields_container.remove_doc(doc_id)

    def _vectorise_tensor_fields_in_batch_per_doc(self, model_config: ModelConfig) -> None:
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
                    vectorisers = Vectoriser.batch_vectorisers_by_modality(model_config, chunks_to_vectorise)

                    for tensor_field_content in doc_field_map[doc_id]:
                        tensor_field_content.vectorise(vectorisers)

                except AddDocumentsError as err:
                    self.add_docs_response_collector.collect_error_response(doc_id, err)
                    self.tensor_fields_container.remove_doc(doc_id)

    def _vectorise_tensor_fields_in_batch_per_add_doc_batch(self, model_config: ModelConfig) -> None:
        with ExitStack() as exit_stack:
            media_repo = self._download_media_contents(exit_stack)
            chunkers = self._field_type_chunker_map(media_repo)
            chunks_map = dict()

            for doc_id, field_name, tensor_field_content in (
                    self.tensor_fields_container.tensor_fields_to_vectorise(*chunkers.keys())):
                try:
                    tensor_field_content.chunk(chunkers)
                    field_type = tensor_field_content.field_type
                    chunks_map.setdefault(field_type, []).extend(tensor_field_content.content_chunks)
                except AddDocumentsError as err:
                    self.add_docs_response_collector.collect_error_response(doc_id, err)
                    self.tensor_fields_container.remove_doc(doc_id)

            try:
                vectorisers = Vectoriser.batch_vectorisers_by_modality(model_config, chunks_map)
            except AddDocumentsError as err:
                logger.error('Encountered problem when vectorising batch of documents. Reason: %s', err)
                raise InternalError(
                    message=f'Encountered problem when vectorising batch of documents. Reason: {str(err)}'
                )

            for doc_id, field_name, tensor_field_content in (
                    self.tensor_fields_container.tensor_fields_to_vectorise(*chunkers.keys())):
                tensor_field_content.vectorise(vectorisers)

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
            media_repo = exit_stack.enter_context(
                add_docs.download_and_preprocess_multimedia_content(
                    docs=list(doc_media_fields.values()),
                    media_field_types_mapping=media_field_types_mapping,
                    marqo_index=self.marqo_index,
                    add_docs_params=self.add_docs_params,
                )
            )

        for url, data in media_repo.items():
            if isinstance(data, Exception):
                for doc_id in url_doc_id_map[url]:
                    self.add_docs_response_collector.collect_error_response(doc_id, AddDocumentsError(
                        error_message=f"Could not process the media file found at `{url}`. Reason: {str(data)}"
                    ))
                    self.tensor_fields_container.remove_doc(doc_id)

        return media_repo

    def _field_type_chunker_map(self, media_repo):
        chunkers: Dict[FieldType, Chunker] = {
            FieldType.Text: TextChunker(text_preprocessing=self.marqo_index.text_preprocessing,
                                        text_chunk_prefix=self.marqo_index.model.get_text_chunk_prefix(
                                            self.add_docs_params.text_chunk_prefix)),
            FieldType.ImagePointer: ImageChunker(media_repo=media_repo,
                                                 image_preprocessing=self.marqo_index.image_preprocessing,
                                                 device=self.add_docs_params.device),
            FieldType.AudioPointer: AudioVideoChunker(media_repo=media_repo),
            FieldType.VideoPointer: AudioVideoChunker(media_repo=media_repo),
        }
        return chunkers

