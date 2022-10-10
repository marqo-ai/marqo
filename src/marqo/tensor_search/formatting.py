"""This module concerns formatting documents to be returned
to users.
"""
from marqo.tensor_search.enums import TensorField
from marqo.tensor_search import utils


def _clean_doc(doc: dict, doc_id=None, include_vectors: bool = False) -> dict:
    """clears tensor search specific fields from the doc

    Args:
        doc: the doc to clean, the doc's "_source" as it's retrieved from marqo-os
        doc_id: if left as None, then the doc will be returned without the _id field
        include_vectors: the vectors will be included with the results
    Returns:

    """
    copied = doc.copy()
    if TensorField.doc_chunk_relation in copied:
        del copied[TensorField.doc_chunk_relation]
    if TensorField.chunk_ids in copied:
        del copied[TensorField.chunk_ids]
    if include_vectors:
        copied["_tensor_components"] = [
            {ch[TensorField.field_name]: ch[TensorField.field_content],
             "_vector": ch[utils.generate_vector_name(ch[TensorField.field_name])]
            } for ch in copied[TensorField.chunks]
        ]
    if TensorField.chunks in copied:
        del copied[TensorField.chunks]
    if doc_id is not None:
        copied['_id'] = doc_id

    return copied
