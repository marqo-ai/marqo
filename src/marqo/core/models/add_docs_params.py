from typing import List
from typing import Optional, Union, Any, Sequence

import numpy as np
from pydantic.v1 import BaseModel, validator, root_validator
from pydantic.v1 import Field

from marqo import marqo_docs
from marqo.api.exceptions import BadRequestError
from marqo.tensor_search.enums import EnvVars
# TODO move deps
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.utils import read_env_vars_and_defaults_ints


class AddDocsParams(BaseModel):
    """Represents the parameters of the document.add_documents() function

    Params:
        index_name: name of the index
        docs: List of documents
        use_existing_tensors: Whether to use the vectors already in doc (for update docs)
        device: Device used to carry out the document update, if `None` is given, it will be determined inference
        image_download_thread_count: number of threads used to concurrently download images
        media_download_headers: headers to authenticate media download requests for audio and video
        mappings: a dictionary used to handle all the object field content in the doc,
            e.g., multimodal_combination field
        model_auth: an object used to authorise downloading an object from a datastore
        text_chunk_prefix: an optional prefix to add to each text chunk
    """

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    # this should only accept Sequences of dicts, but currently validation lies elsewhere
    docs: Union[Sequence[Union[dict, Any]], np.ndarray]

    index_name: str
    device: Optional[str]
    tensor_fields: Optional[List] = Field(default_factory=None)
    image_download_thread_count: Optional[int] = None
    media_download_thread_count: Optional[int] = None
    media_download_headers: Optional[dict] = None
    use_existing_tensors: bool = False
    mappings: Optional[dict] = None
    model_auth: Optional[ModelAuth] = None
    text_chunk_prefix: Optional[str] = None

    def __init__(self, **data: Any):
        super().__init__(**data)

    @root_validator(pre=True)
    def validate_thread_counts(cls, values):
        """
        Set the values for image_download_thread_count and media_download_thread_count.
        There are 4 cases:
            1. Both not given -> Both reads default values
            2. Image given, media not given -> media reads default value, image uses given value
            3. Media set, image not set -> media uses given value, image uses media value
            4. Both set -> error:
        Once set, media_download_thread_count is used for audio and video, image_download_thread_count is
        used for images, when sending inference requests to the inference server.
        """
        image_count = values.get('image_download_thread_count')
        media_count = values.get('media_download_thread_count')
        if media_count and image_count:
            raise ValueError("Cannot set both 'image_download_thread_count' and 'media_download_thread_count'.")
        elif image_count is None and media_count is None:
            # Set default values for both
            values['image_download_thread_count'] = (
                read_env_vars_and_defaults_ints(EnvVars.MARQO_IMAGE_DOWNLOAD_THREAD_COUNT_PER_REQUEST))
            values['media_download_thread_count'] = (
                read_env_vars_and_defaults_ints(EnvVars.MARQO_MEDIA_DOWNLOAD_THREAD_COUNT_PER_REQUEST)
            )
        elif image_count is None and media_count is not None:
            values['media_download_thread_count'] = media_count
            values['image_download_thread_count'] = media_count
        elif image_count is not None and media_count is None:
            values['media_download_thread_count'] = (
                read_env_vars_and_defaults_ints(EnvVars.MARQO_MEDIA_DOWNLOAD_THREAD_COUNT_PER_REQUEST)
            )
            values['image_download_thread_count'] = image_count
        else:
            raise ValueError("Invalid combination of image_download_thread_count and media_download_thread_count.")
        return values

    @validator('docs')
    def validate_docs(cls, docs):
        doc_count = len(docs)

        max_doc = read_env_vars_and_defaults_ints(EnvVars.MARQO_MAX_DOCUMENTS_BATCH_SIZE)

        if doc_count == 0:
            raise BadRequestError(message="Received empty add documents request")
        elif doc_count > max_doc:
            raise BadRequestError(
                message=f"Number of docs in add documents request ({doc_count}) exceeds limit of {max_doc}. "
                        f"If using the Python client, break up your `add_documents` request into smaller batches using "
                        f"its `client_batch_size` parameter. "
                        f"See {marqo_docs.api_reference_document_body()} for more details."
            )

        return docs