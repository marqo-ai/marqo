import numpy as np
from numpy import ndarray

from tritonclient import grpc
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
        self.client = grpc.InferenceServerClient(url=self.url, verbose=False)

    def encode(self, inputs: ndarray, modality: Modality) -> ndarray:
        """
        Sends a request to the Triton inference server to encode the input data.

        :param inputs: The input data to be encoded, as a numpy array.
        :return: The encoded output from the Triton server, as a numpy array.
        """
        if modality == Modality.TEXT:
            inputs = np.ascontiguousarray(inputs, dtype=np.int32)
            data_type = "INT32"
        elif modality == Modality.IMAGE:
            inputs = np.ascontiguousarray(inputs, dtype=np.float32)
            data_type = "FP32"
        else:
            raise ValueError(f"Unsupported modality: {modality}. Supported modalities are TEXT and IMAGE.")
        model_name=self._get_model_name(modality)

        logger.info(f"Encoding processed - {model_name}")

        input_tensor = grpc.InferInput("input", list(inputs.shape), data_type)
        input_tensor.set_data_from_numpy(inputs)
        output_tensor = grpc.InferRequestedOutput("output")
        result = self.client.infer(
            model_name=model_name,
            inputs=[input_tensor],
            outputs=[output_tensor]
        )
        output_data = result.as_numpy("output")
        return output_data

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