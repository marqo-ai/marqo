import marqo

from typing import List, Dict, Any


def index_transcriptions(
    annotated_transcriptions: List[Dict[str, Any]],
    index: str,
    mq: marqo.Client,
    tensor_fields: List[str] = [],
    device: str = "cpu",
    batch_size: int = 32,
) -> Dict[str, str]:

    # drop short transcriptions and transcriptions that consist of duplicated repeating
    # character artifacts
    annotated_transcriptions = [
        at
        for at in annotated_transcriptions
        if len(at["transcription"]) > 5 or len({*at["transcription"]}) > 4
    ]

    response = mq.index(index).add_documents(
        annotated_transcriptions,
        tensor_fields=tensor_fields,
        device=device,
        client_batch_size=batch_size
    )

    return response
