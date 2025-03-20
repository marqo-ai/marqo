import httpx

from marqo import logging
from marqo.core.inference.api import Inference, InferenceResult, InferenceRequest, InferenceError

import msgpack
import msgpack_numpy

msgpack_numpy.patch()


logger = logging.get_logger(__name__)


class NativeInferenceClient(Inference):
    def __init__(self, base_url: str):
        """
        Args:
            base_url (str): The base URL of the remote inference service.
        """
        self.base_url = base_url.rstrip("/")

        # TODO is default connection pooling config good enough, or do we want to config the limit here?
        #   limits=httpx.Limits(max_keepalive_connections=?, max_connections=?)
        # TODO set proper timeout
        # TODO see if retry is needed
        self.client = httpx.Client(base_url=base_url)

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
                except Exception as parse_error:
                    logger.warning(f'Error parsing error message: {str(parse_error)}')
                    error_message = 'Error parsing error message in msgpack format'
            else:
                error_message = str(e)
            raise InferenceError(f"HTTP error when calling remote inference service: {error_message}") from e

        # Unpack the MessagePack response (with numpy support)
        try:
            result_dict = msgpack.unpackb(response.content, raw=False)
            return InferenceResult.parse_obj(result_dict)
        except Exception as e:
            raise InferenceError(f"Error decoding MessagePack response: {str(e)}") from e
