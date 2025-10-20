from tritonclient import grpc

from inference_orchestrator.core.logging import get_logger
from inference_orchestrator.schemas.triton_channel_args import TritonChannelArgs
from inference_orchestrator.services.errors import TritonInferenceError

logger = get_logger(__name__)


class TritonGRPCClient:
    def __init__(self, url: str, triton_channel_args: TritonChannelArgs):
        parsed_url = self._parse_url(url)
        logger.info(
            f"Instantiating Triton GRPC client with URL: {parsed_url} and channel args: {triton_channel_args}"
        )
        self.client = grpc.InferenceServerClient(
            url=parsed_url,
            verbose=False,
            channel_args=triton_channel_args.build_channel_args(),
        )
        self.grpc_compression_algorithm = triton_channel_args.grpc_compression_algorithm

    def close(self):
        """
        Close the gRPC client connection.
        """
        logger.info("Closing the gRPC client connection")
        self.client.close()

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
        url = url.removeprefix("https://").removeprefix("http://")
        return url

    def encode(
        self,
        model_name: str,
        infer_inputs: list[grpc.InferInput],
        infer_outputs: list[grpc.InferRequestedOutput],
    ) -> grpc.InferResult:
        """
        Encode the input data using the specified model.

        Args:
            model_name: The name of the model to be used for encoding.
            infer_inputs: A list of infer input data.
            infer_outputs: A list of infer output data
        Returns:
            The gRPC inference result containing the encoded data.
        Raises:
            TritonInferenceError: If there is an error during the inference process.
        """
        try:
            return self.client.infer(
                model_name=model_name,
                inputs=infer_inputs,
                outputs=infer_outputs,
                compression_algorithm=self.grpc_compression_algorithm,
            )
        except grpc.InferenceServerException as e:
            raise TritonInferenceError(
                f"Error during inference with model {model_name}: {str(e)}"
            ) from e
