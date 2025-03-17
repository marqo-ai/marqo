import torch
from torch import Tensor

from marqo.inference.native_inference.embedding_models.abstract_embedding_model import AbstractEmbeddingModel
from marqo.inference.type import *


def encode_processed_content(model, preprocessed_content_list: list[PreprocessedContent],
                             modality, normalize, maximum_batch_size=16) -> list[Tensor]:
    """
    Encode the processed content using the model.

    Args:
        model (AbstractEmbeddingModel): The model to use for encoding.
        preprocessed_content_list (list[PreprocessedContent]): The processed content to encode.
        modality (Modality): The modality of the content.
        normalize (bool): Whether to normalize the embeddings.

    Returns:
        list[Tensor]: The embeddings of the processed content.
    """

    flattened_content: List[Tensor] = _collect_tensors(preprocessed_content_list)
    if len(flattened_content) > 0:
        embeddings = []
        stacked_content = torch.cat(flattened_content)
        for i in range(0, len(stacked_content), maximum_batch_size):
            batch = stacked_content[i:i + maximum_batch_size]
            batch_embeddings = model.encode(batch, modality, normalize)
            embeddings.extend(batch_embeddings)

        if len(embeddings) != len(flattened_content):
            raise ValueError("The number of embeddings does not match the number of contents")

        return embeddings
    else:
        return []


def _collect_tensors(preprocessed_content: list[list[tuple[str, Tensor]]]) -> list[Tensor]:
    collected_tensors = []
    for chunk in preprocessed_content:
        if isinstance(chunk, list):
            for _, tensor in chunk:
                if isinstance(tensor, Tensor):
                    collected_tensors.append(tensor)
        elif isinstance(chunk, (MediaDownloadError, PreprocessingError)):
            continue
    return collected_tensors


def format_results(preprocessed_content_list: list[PreprocessedContent], embeddings) \
        -> InferenceResult:
    results = []
    embedding_index = 0
    for chunk in preprocessed_content_list:
        chunk_results = []
        if isinstance(chunk, (MediaDownloadError, PreprocessingError)):
            results.append(chunk)
            continue
        elif isinstance(chunk, list):
            for original_text, chunk_content in chunk:
                chunk_results.append((original_text, embeddings[embedding_index]))
                embedding_index += 1
        else:
            raise ValueError(f"Invalid chunk type: {type} for chunk: {chunk}")
        results.append(chunk_results)
    if len(results) != len(preprocessed_content_list):
        raise ValueError("The formatted results length does not match the input content length")
    return InferenceResult(result=results)






