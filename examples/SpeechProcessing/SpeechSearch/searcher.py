import marqo
import numpy as np
from typing import List, Dict, Union, Callable, Tuple, Any


def search_transcripts(
    query: str,
    limit: int,
    index: str,
    mq: marqo.Client,
    audio_getter: Callable[[str, int], np.ndarray],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Union[str, np.ndarray]]]]:
    print("Searching...")
    results = mq.index(index).search(
        q=query,
        limit=limit,
    )
    print("Done!")

    segments = []
    print("Extracting audio for hits...")
    for result in results["hits"]:
        start = int(result["start"] * result["samplerate"])
        end = int(result["end"] * result["samplerate"])

        audio = audio_getter(result["file"], result["samplerate"])

        segment = {
            "transcription": result["transcription"],
            "audio": audio[start:end],
            "samplerate": result["samplerate"],
        }
        segments.append(segment)
    print("Done!")
    return results, segments
