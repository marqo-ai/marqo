from inference_orchestrator.core.logging import get_logger
from inference_orchestrator.services.triton_inference.triton.channel_args import (
    ChannelArgs,
)
from tritonclient import grpc

logger = get_logger(__name__)


class TritonGRPCClient:
    def __init__(self, url: str, channel_args: ChannelArgs):
        parsed_url = self._parse_url(url)
        logger.info(
            f"Instantiating Triton GRPC client with URL: {parsed_url} and channel args: {channel_args}"
        )
        self.client = grpc.InferenceServerClient(
            url=parsed_url,
            verbose=False,
            channel_args=channel_args.build_channel_args(),
        )
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
        """

        return self.client.infer(
            model_name=model_name,
            inputs=infer_inputs,
            outputs=infer_outputs,
            compression_algorithm=self.grpc_compression_algorithm,
        )
