from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import Optional
from marqo.tensor_search.models.external_apis.abstract_classes import ExternalAuth
from typing import List


@dataclass(frozen=True)
class AddDocsParams:
    """Represents the parameters of the tensor_search.add_documents() function

    Params:
        index_name: name of the index
        docs: List of documents
        auto_refresh: Set to False if indexing lots of docs
        non_tensor_fields: List of fields, within documents to not create tensors for. Default to
          make tensors for all fields.
        use_existing_tensors: Whether to use the vectors already in doc (for update docs)
        device: Device used to carry out the document update.
        update_mode: {'replace' | 'update'}. If set to replace (default) just
        image_download_thread_count: number of threads used to concurrently download images
        image_download_headers: headers to authenticate image download
        mappings: a dictionary used to handle all the object field content in the doc, e.g., multimodal_combination field

    """
    index_name: str
    docs: List[dict]
    auto_refresh: bool
    non_tensor_fields: List = Field(default_factory=list)
    device: Optional[str] = None
    update_mode: str = "replace"
    image_download_thread_count: int = 20
    image_download_headers: dict = Field(default_factory=dict)
    use_existing_tensors: bool = False
    mappings: Optional[dict] = None
