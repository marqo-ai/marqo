from tritonclient import grpc
import numpy as np

from numpy import ndarray
from marqo.inference.triton_inference.triton.input_type import InputType
from marqo.inference.triton_inference.triton.channel_args import ChannelArgs
from marqo.logging import get_logger


logger = get_logger(__name__)


class TritonGRPCClient:
    def __init__(self, url: str, channel_args: ChannelArgs):
        parsed_url = self._parse_url(url)
        logger.info(f"Instantiating Triton GRPC client with URL: {parsed_url} and channel args: {channel_args}")
        self.client = grpc.InferenceServerClient(url=parsed_url, verbose=False, channel_args=channel_args.build_channel_args())
        self.grpc_compression_algorithm = channel_args.grpc_compression_algorithm

    def _parse_url(self, url):
        """
        Remove the prefix http:// or https:// from the provided url if present.

        Args:
            url: The URL string to be parsed.

        Returns:
            url: The parsed URL.
        """
        if not url:
            raise ValueError("The triton server URL cannot be empty.")
        if "http://" in url:
            url = url.replace("http://", "")
        if "https://" in url:
            url = url.replace("https://", "")
        return url

    def encode(self, model_name: str, inputs: ndarray, input_type: InputType) -> ndarray:
        """
        Encode the input data using the specified model.

        Args:
            model_name: The name of the model to be used for encoding.
            inputs: The input data to be encoded.
            input_type: The type of the input data.

        Returns:
            The encoded output data.
        """
        inputs = np.ascontiguousarray(inputs, dtype=input_type.dtype)

        input_tensor = grpc.InferInput(name="input", shape=list(inputs.shape), datatype=input_type.code)
        input_tensor.set_data_from_numpy(inputs)

        output_tensor = grpc.InferRequestedOutput(name="output")
        response = self.client.infer(
            model_name=model_name, inputs=[input_tensor], outputs=[output_tensor],
            compression_algorithm=self.grpc_compression_algorithm
        )
        output_data = response.as_numpy("output")
        if output_data.shape[0] != inputs.shape[0]:
            raise ValueError(f"Output data shape {output_data.shape} does not match input data shape {inputs.shape}")
        return output_data