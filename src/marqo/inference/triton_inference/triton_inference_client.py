import numpy as np
from numpy import ndarray

import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc
from marqo.core.inference.api import Modality
from marqo.logging import get_logger
from timeit import default_timer as timer

logger = get_logger(__name__)


class TritonInferenceClient:
    """
    A client for interacting with a Triton inference server.
    """
    def __init__(self, url: str):
        """
        Initializes the TritonInferenceClient with the server URL and model name.

        :param url: The URL of the Triton inference server.
        """
        self.url = url
        channel = grpc.insecure_channel(url.split("://")[1], compression=grpc.Compression.Gzip)
        self.grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    def encode(self, inputs: ndarray, modality: Modality) -> ndarray:
        """
        Sends a request to the Triton inference server to encode the input data.

        :param inputs: The input data to be encoded, as a numpy array.
        :return: The encoded output from the Triton server, as a numpy array.
        """
        inference_start_time = timer()
        request = service_pb2.ModelInferRequest(
            model_name=self._get_model_name(modality),  # Replace with your actual model name
            inputs=[self._get_input(inputs, modality)],
            outputs=[
                service_pb2.ModelInferRequest.InferRequestedOutputTensor(name="output")
            ]
        )
        duration = timer() - inference_start_time

        logger.info(f"Prepared inference request in {round(duration * 1000)} ms")
        start_time = timer()
        response = self.grpc_stub.ModelInfer(request)
        embeddings = np.frombuffer(response.raw_output_contents[0], dtype=np.float32).reshape(inputs.shape[0], -1)
        duration = timer() - start_time
        logger.info(f"Inference took {round(duration * 1000)} ms")
        return embeddings

    def _get_model_name(self, modality: Modality) -> str:
        """
        Returns the model name based on the modality.

        :param modality: The modality of the input data.
        :return: The model name as a string.
        """
        if modality == Modality.TEXT:
            return "ViT-B-16-SigLI-FN-Text"
        elif modality == Modality.IMAGE:
            return "ViT-B-16-SigLI-FN-Image"
        else:
            raise ValueError(f"Unsupported modality: {modality}. Supported modalities are TEXT and IMAGE.")

    def _get_input_type(self, modality: Modality) -> str:
        """
        Returns the input type based on the modality.

        :param modality: The modality of the input data.
        :return: The input type as a string.
        """
        if modality == Modality.TEXT:
            return "INT32"
        elif modality == Modality.IMAGE:
            return "FP32"
        else:
            raise ValueError(f"Unsupported modality: {modality}. Supported modalities are TEXT and IMAGE.")

    def _get_input(self, inputs: ndarray, modality: Modality) -> service_pb2.ModelInferRequest.InferInputTensor:
        """
        Creates an InferInputTensor for the given inputs and modality.

        :param inputs: The input data to be sent to the Triton server.
        :param modality: The modality of the input data.
        :return: An InferInputTensor object containing the input data.
        """
        if modality == Modality.TEXT:
            return service_pb2.ModelInferRequest.InferInputTensor(
                name="input",
                datatype="INT32",
                shape=list(inputs.shape),
                contents=service_pb2.InferTensorContents(
                    int_contents=inputs.flatten().astype(np.int32).tolist()
                )
            )
        elif modality == Modality.IMAGE:
            return service_pb2.ModelInferRequest.InferInputTensor(
                name="input",
                datatype="FP32",
                shape=list(inputs.shape),
                contents=service_pb2.InferTensorContents(
                    fp32_contents=inputs.flatten().astype(np.float32).tolist()
                ),
            )
        else:
            raise ValueError(f"Unsupported modality: {modality}. Supported modalities are TEXT and IMAGE.")
