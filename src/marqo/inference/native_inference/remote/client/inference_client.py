import httpx
import pydantic
from httpx import Timeout

from marqo import logging
from marqo.core.inference.api import Inference, InferenceResult, InferenceRequest, InferenceError

import msgpack
import msgpack_numpy

msgpack_numpy.patch()


logger = logging.get_logger(__name__)


class NativeInferenceClient(Inference):
    def __init__(self, base_url: str, pool_size: int = 20, timeout: int = 300):
        """
        Args:
            base_url (str): The base URL of the remote inference service.
            pool_size (int): The connection pool size of the httpx client. Please note that we don't cap the active
                connection number. So if throttling limit is raised, the requests to inference server will not queue up.
            timeout (int): The timeout of httpx client to inference server in seconds. We set a large default timeout
                to cater for batch inference requests containing multiple images or media files
        """
        self.base_url = base_url.rstrip("/")
        limits = httpx.Limits(max_keepalive_connections=pool_size, max_connections=None)
        self.client = httpx.Client(base_url=base_url, limits=limits, timeout=Timeout(timeout=timeout))

    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        """
        Sends the inference request encoded with MessagePack (using msgpack_numpy) to the remote FastAPI endpoint
        and returns the deserialized inference result.
        """
        url = f"{self.base_url}/vectorise"
        headers = {"Content-Type": "application/msgpack", "Accept": "application/msgpack"}

        # Convert the request to a dict (honoring aliases) and then pack with MessagePack
        request_dict = request.dict(by_alias=True)
        request_bytes = msgpack.packb(request_dict, use_bin_type=True)

        try:
            response = self.client.post(url, headers=headers, content=request_bytes)
            response.raise_for_status()
        except httpx.HTTPError as e:
            # The error response is also msgpack encoded
            if isinstance(e, httpx.HTTPStatusError) and e.response is not None and e.response.content:
                try:
                    error_response = msgpack.unpackb(e.response.content, raw=False)
                    error_message = error_response["detail"]
                except (msgpack.ExtraData, msgpack.UnpackException, msgpack.UnpackValueError):
                    logger.warning('Error parsing error message', exc_info=True)
                    error_message = 'Error parsing error message in msgpack format'
            else:
                error_message = str(e)
            raise InferenceError(f"HTTP error when calling remote inference service: {error_message}") from e

        # Unpack the MessagePack response (with numpy support)
        try:
            result_dict = msgpack.unpackb(response.content, raw=False)
            return InferenceResult.parse_obj(result_dict)
        except (msgpack.ExtraData, msgpack.UnpackException, msgpack.UnpackValueError, pydantic.ValidationError) as e:
            raise InferenceError(f"Error decoding MessagePack response: {str(e)}") from e
