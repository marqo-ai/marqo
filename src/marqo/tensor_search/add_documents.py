import copy
import datetime
from timeit import default_timer as timer
import uuid
from typing import List, Optional, Union, Iterable, Sequence, Dict, Any
from PIL import Image
from marqo.tensor_search.enums import (
    MediaType, MlModel, TensorField, SearchMethod, OpenSearchDataType,
    EnvVars
)
from marqo.tensor_search.enums import IndexSettingsField as NsField
from marqo.tensor_search import utils, validation, configs, parallel
from marqo.s2_inference.processing import text as text_processor
from marqo.s2_inference.processing import image as image_processor
from marqo.s2_inference.clip_utils import _is_image
from marqo.s2_inference import s2_inference
from marqo import errors
from marqo.s2_inference import errors as s2_inference_errors
import threading
from typing import NamedTuple, Tuple
from marqo.tensor_search.tensor_search_logging import get_logger

logger = get_logger(__name__)


class UnsuccessfulTensorise(NamedTuple):
    # refers to the position of the doc in the add_documents call
    error_details: dict


class SuccessfulTensoriseOutput(NamedTuple):
    indexing_instruction: dict
    index_ready_doc: dict

    def to_os_instructions(self) -> List[Dict]:
        """Returns a list of the indexing instruction followed by the content to be indexed"""
        return [self.indexing_instruction, self.index_ready_doc]


class DocAsIndexingInstruction(NamedTuple):
    vectorise_time: float
    new_fields: set
    doc_pos: int
    tensorised_for_indexing: SuccessfulTensoriseOutput = None
    failure_details: Optional[UnsuccessfulTensorise] = None


def threaded_docs_to_instructions(
        enumerated_docs: List[Tuple[int, dict]], update_mode: str, index_name: str, existing_fields,
        non_tensor_fields: dict, index_info, selected_device,

        vectorise_times: list, new_fields_lock: threading.Lock, new_fields: set,
        unsuccessful_docs: list, to_be_indexed: list
    ) -> None:
    """"""
    for doc_pos, doc in enumerated_docs:
        threaded_doc_to_instructions(
            doc_pos=doc_pos, doc=doc, update_mode=update_mode, index_name=index_name,
            existing_fields=existing_fields, non_tensor_fields=non_tensor_fields, index_info=index_info,
            selected_device=selected_device, vectorise_times=vectorise_times, new_fields_lock=new_fields_lock,
            new_fields=new_fields, unsuccessful_docs=unsuccessful_docs, to_be_indexed=to_be_indexed
        )


def threaded_doc_to_instructions(
        doc: dict, update_mode: str, index_name: str, doc_pos: int, existing_fields,
        non_tensor_fields: dict, index_info, selected_device,

        vectorise_times: list, new_fields_lock: threading.Lock, new_fields: set,
        unsuccessful_docs: list, to_be_indexed: list
    ) -> None:
    indexing_instructions = doc_to_indexing_instructions(
        doc_pos=doc_pos, doc=doc, update_mode=update_mode, index_name=index_name,
        existing_fields=existing_fields,
        non_tensor_fields=non_tensor_fields, index_info=index_info, selected_device=selected_device
    )
    vectorise_times[doc_pos] = indexing_instructions.vectorise_time
    with new_fields_lock:
        for f in indexing_instructions.new_fields:
            new_fields.add(f)

    if indexing_instructions.failure_details is not None:
        unsuccessful_docs.append(
            (indexing_instructions.doc_pos, indexing_instructions.failure_details.error_details))
    else:
        # no failures encountered
        if indexing_instructions.tensorised_for_indexing is not None:
            to_be_indexed[doc_pos] = indexing_instructions.tensorised_for_indexing.to_os_instructions()


def doc_to_indexing_instructions(
        doc: dict, update_mode: str, index_name: str, doc_pos: int, existing_fields, non_tensor_fields: dict,
        index_info, selected_device
    ) -> DocAsIndexingInstruction:
    """From a customer-created doc

    Returns:
        Tuple of dicts:
    """
    indexing_instructions = {'index' if update_mode == 'replace' else 'update': {"_index": index_name}}
    copied = copy.deepcopy(doc)

    document_is_valid = True
    new_fields_from_doc = set()
    doc_vectorise_time = 0

    doc_id = None
    try:
        validation.validate_doc(doc)

        if "_id" in doc:
            doc_id = validation.validate_id(doc["_id"])
            del copied["_id"]
        else:
            doc_id = str(uuid.uuid4())

        [validation.validate_field_name(field) for field in copied]
    except errors.__InvalidRequestError as err:
        return DocAsIndexingInstruction(
            doc_pos=doc_pos,
            new_fields=new_fields_from_doc,
            vectorise_time=doc_vectorise_time,
            failure_details=UnsuccessfulTensorise(
                error_details={'_id': doc_id if doc_id is not None else '',
                             'error': err.message, 'status': int(err.status_code), 'code': err.code}
                ),
            )

    if update_mode == "replace":
        indexing_instructions["index"]["_id"] = doc_id
    else:
        indexing_instructions["update"]["_id"] = doc_id

    chunks = []

    for field in copied:

        try:
            field_content = validation.validate_field_content(copied[field])
        except errors.InvalidArgError as err:
            document_is_valid = False
            return DocAsIndexingInstruction(
                doc_pos=doc_pos,
                new_fields=set(),
                vectorise_time=doc_vectorise_time,
                failure_details=UnsuccessfulTensorise(
                    error_details={
                        '_id': doc_id, 'error': err.message, 'status': int(err.status_code),
                        'code': err.code}
                )
            )

        if field not in existing_fields:
            new_fields_from_doc.add((field, utils.infer_opensearch_data_type(copied[field])))

        # Don't process text/image fields when explicitly told not to.
        if field in non_tensor_fields:
            continue

        # TODO put this into a function to determine routing
        if isinstance(field_content, (str, Image.Image)):

            # TODO: better/consistent handling of a no-op for processing (but still vectorize)

            # 1. check if urls should be downloaded -> "treat_pointers_and_urls_as_images":True
            # 2. check if it is a url or pointer
            # 3. If yes in 1 and 2, download blindly (without type)
            # 4. Determine media type of downloaded
            # 5. load correct media type into memory -> PIL (images), videos (), audio (torchaudio)
            # 6. if chunking -> then add the extra chunker

            if isinstance(field_content, str) and not _is_image(field_content):

                split_by = index_info.index_settings[NsField.index_defaults][NsField.text_preprocessing][
                    NsField.split_method]
                split_length = index_info.index_settings[NsField.index_defaults][NsField.text_preprocessing][
                    NsField.split_length]
                split_overlap = index_info.index_settings[NsField.index_defaults][NsField.text_preprocessing][
                    NsField.split_overlap]
                content_chunks = text_processor.split_text(field_content, split_by=split_by,
                                                           split_length=split_length, split_overlap=split_overlap)
                text_chunks = content_chunks
            else:
                # TODO put the logic for getting field parameters into a function and add per field options
                image_method = index_info.index_settings[NsField.index_defaults][NsField.image_preprocessing][
                    NsField.patch_method]
                # the chunk_image contains the no-op logic as of now - method = None will be a no-op
                try:
                    # in the future, if we have different chunking methods, make sure we catch possible
                    # errors of different types generated here, too.
                    content_chunks, text_chunks = image_processor.chunk_image(
                        field_content, device=selected_device, method=image_method)
                except s2_inference_errors.S2InferenceError:
                    document_is_valid = False
                    image_err = errors.InvalidArgError(message=f'Could not process given image: {field_content}')
                    return DocAsIndexingInstruction(
                        vectorise_time=doc_vectorise_time,
                        doc_pos=doc_pos,
                        new_fields=new_fields_from_doc,
                        failure_details=UnsuccessfulTensorise(error_details={
                            '_id': doc_id, 'error': image_err.message, 'status': int(image_err.status_code),
                            'code': image_err.code})
                    )

            normalize_embeddings = index_info.index_settings[NsField.index_defaults][
                NsField.normalize_embeddings]
            infer_if_image = index_info.index_settings[NsField.index_defaults][
                NsField.treat_urls_and_pointers_as_images]

            try:
                # in the future, if we have different underlying vectorising methods, make sure we catch possible
                # errors of different types generated here, too.

                # ADD DOCS TIMER-LOGGER (4)
                start_time = timer()
                vector_chunks = s2_inference.vectorise(model_name=index_info.model_name,
                                                       model_properties=utils.get_model_properties(index_info),
                                                       content=content_chunks,
                                                       device=selected_device,
                                                       normalize_embeddings=normalize_embeddings,
                                                       infer=infer_if_image)

                end_time = timer()
                single_vectorise_call = end_time - start_time
                # TODO TODO TODO return single vectorise call, to be added in the calling function
                doc_vectorise_time += single_vectorise_call
                logger.debug(f"(4) TIME for single vectorise call: {(single_vectorise_call):.3f}s.")
            except (s2_inference_errors.UnknownModelError,
                    s2_inference_errors.InvalidModelPropertiesError,
                    s2_inference_errors.ModelLoadError) as model_error:
                raise errors.BadRequestError(
                    message=f'Problem vectorising query. Reason: {str(model_error)}',
                    link="https://marqo.pages.dev/latest/Models-Reference/dense_retrieval/"
                )
            except s2_inference_errors.S2InferenceError:
                document_is_valid = False
                image_err = errors.InvalidArgError(message=f'Could not process given image: {field_content}')
                return DocAsIndexingInstruction(
                    vectorise_time=doc_vectorise_time,
                    new_fields=new_fields_from_doc,
                    doc_pos=doc_pos,
                    failure_details=UnsuccessfulTensorise(error_details={
                        '_id': doc_id, 'error': image_err.message, 'status': int(image_err.status_code),
                         'code': image_err.code})
                )

            if (len(vector_chunks) != len(text_chunks)):
                raise RuntimeError(
                    f"the input content after preprocessing and its vectorized counterparts must be the same length."
                    f"recevied text_chunks={len(text_chunks)} and vector_chunks={len(vector_chunks)}. "
                    f"check the preprocessing functions and try again. ")

            for text_chunk, vector_chunk in zip(text_chunks, vector_chunks):
                # only add chunk values which are string, boolean or numeric
                chunk_values_for_filtering = {}
                for key, value in copied.items():
                    if not (isinstance(value, str) or isinstance(value, float)
                            or isinstance(value, bool) or isinstance(value, int)):
                        continue
                    chunk_values_for_filtering[key] = value
                chunks.append({
                    utils.generate_vector_name(field): vector_chunk,
                    TensorField.field_content: text_chunk,
                    TensorField.field_name: field,
                    **chunk_values_for_filtering
                })
    if document_is_valid:
        # TODO TODO TODO this must be added to the parent function
        # new_fields = new_fields.union(new_fields_from_doc)

        if update_mode == 'replace':
            copied[TensorField.chunks] = chunks

            formatted = SuccessfulTensoriseOutput(indexing_instruction=indexing_instructions, index_ready_doc=copied)
            return DocAsIndexingInstruction(
                vectorise_time=doc_vectorise_time,
                doc_pos=doc_pos,
                tensorised_for_indexing=formatted,
                new_fields=new_fields_from_doc,
            )
        else:
            to_upsert = copied.copy()
            to_upsert[TensorField.chunks] = chunks

            doc_for_upserting = {
                "upsert": to_upsert,
                "script": {
                    "lang": "painless",
                    "source": f"""

        // updates the doc's fields with the new content
        for (key in params.customer_dict.keySet()) {{
            ctx._source[key] = params.customer_dict[key];
        }}            

        // keep track of the merged doc
        def merged_doc = [:];
        merged_doc.putAll(ctx._source);
        merged_doc.remove("{TensorField.chunks}");

        // remove chunks if the __field_name matches an updated field
        // All update fields should be recomputed, and it should be safe to delete these chunks             
        for (int i=ctx._source.{TensorField.chunks}.length-1; i>=0; i--) {{
            if (params.doc_fields.contains(ctx._source.{TensorField.chunks}[i].{TensorField.field_name})) {{
               ctx._source.{TensorField.chunks}.remove(i);
            }}
            // Check if the field should have a tensor, remove if not.
            else if (params.non_tensor_fields.contains(ctx._source.{TensorField.chunks}[i].{TensorField.field_name})) {{
                ctx._source.{TensorField.chunks}.remove(i);
            }}
        }}

        // update the chunks, setting fields to the new data
        for (int i=ctx._source.{TensorField.chunks}.length-1; i>=0; i--) {{
            for (key in params.customer_dict.keySet()) {{
                ctx._source.{TensorField.chunks}[i][key] = params.customer_dict[key];
            }}
        }}

        // update the new chunks, adding the existing data 
        for (int i=params.new_chunks.length-1; i>=0; i--) {{
            for (key in merged_doc.keySet()) {{
                params.new_chunks[i][key] = merged_doc[key];
            }}
        }}

        // appends the new chunks to the existing chunks  
        ctx._source.{TensorField.chunks}.addAll(params.new_chunks);

                    """,
                    "params": {
                        "doc_fields": list(copied.keys()),
                        "new_chunks": chunks,
                        "customer_dict": copied,
                        "non_tensor_fields": non_tensor_fields
                    },
                }
            }
            formatted = SuccessfulTensoriseOutput(indexing_instruction=indexing_instructions,
                                                  index_ready_doc=doc_for_upserting)
            return DocAsIndexingInstruction(
                vectorise_time=doc_vectorise_time,
                doc_pos=doc_pos,
                tensorised_for_indexing=formatted,
                new_fields=new_fields_from_doc,
            )
