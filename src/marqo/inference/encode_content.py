import torch

from marqo.core.inference.api import *
from marqo.inference.native_inference.embedding_models.abstract_embedding_model import AbstractEmbeddingModel
from torch import Tensor


def encode_processed_content(model, preprocessed_content: list[list[tuple[str, Tensor]]],
                             modality, normalize) -> list[Tensor]:
    """
    Encode the processed content using the model.

    Args:
        model (AbstractEmbeddingModel): The model to use for encoding.
        processed_content (list[tuple[Tensor]]): The processed content to encode.
        modality (Modality): The modality of the content.
        normalize (bool): Whether to normalize the embeddings.

    Returns:
        list[Tensor]: The embeddings of the processed content.
    """
    flattened_content: List[Tensor] = [tensor for chunk in preprocessed_content for _, tensor in chunk]
    stacked_content = torch.cat(flattened_content)
    embeddings = model.encode(stacked_content, modality, normalize)
    return embeddings


def format_results(preprocessed_content, embeddings) -> InferenceResult:
    results = []
    embedding_index = 0
    for chunk in preprocessed_content:
        chunk_results = []
        for original_text, _ in chunk:
            chunk_results.append((original_text, embeddings[embedding_index]))
            embedding_index += 1
        results.append(chunk_results)
    return InferenceResult(result=results)






