import numpy as np
import librosa
from typing import Dict, List, Union, ImageType, Tuple, FloatTensor, ndarray, Literal
from marqo.s2_inference.clap_utils import load_audio_from_path


def chunk_audio(
    audio: Union[str, np.ndarray],
    method: Literal["simple", "energy", None],
    sample_rate: int = 48000,
    chunk_length_sec: int = 10,
) -> Tuple[List[np.ndarray], np.ndarray]:

    if method in [None, "none", "", "None", " "]:
        if isinstance(audio, str):
            return [audio], [audio]
        elif isinstance(audio, np.ndarray):
            return [audio], [(0, len(audio) / sample_rate)]
        else:
            raise TypeError(
                f"only pointers to an audio file or a numpy array are allowed. received {type(audio)}"
            )

    if isinstance(audio, str):
        audio_array = load_audio_from_path(audio)
    else:
        audio_array = audio

    if method == "simple":
        return simple_audio_chunker(audio_array, sample_rate, chunk_length_sec)
    elif method == "energy":
        return energy_based_chunker(audio_array, sample_rate, chunk_length_sec)

    raise ValueError(f"Invalid audio chunking method: {method}")


def simple_audio_chunker(
    audio_array: np.ndarray, sample_rate: int, chunk_length_sec: int
) -> Tuple[List[np.ndarray], np.ndarray]:
    chunk_size = chunk_length_sec * sample_rate
    num_chunks = int(np.ceil(len(audio_array) / chunk_size))
    chunks = []
    timestamps = np.zeros((num_chunks, 2))

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunks.append(audio_array[start_idx:end_idx])
        timestamps[i] = [
            start_idx / sample_rate,
            min(len(audio_array), end_idx) / sample_rate,
        ]

    return chunks, timestamps


def _calculate_energy(
    y: np.ndarray, frame_length: int = 1024, hop_length: int = 512
) -> np.ndarray:
    """Calculate the energy of an audio signal"""
    energy = np.array(
        [
            np.sum(np.abs(y[i : i + frame_length] ** 2))
            for i in range(0, len(y), hop_length)
        ]
    )
    return energy


def _find_energy_peaks(
    energy: np.ndarray, threshold: float, sample_rate: int, hop_length: int = 512
) -> np.ndarray:
    peaks = np.where(energy > threshold)[0]
    times = librosa.frames_to_time(peaks, sr=sample_rate, hop_length=hop_length)
    return times


def _enegery_audio_segmenter(
    audio_array: np.ndarray,
    sample_rate: int,
    min_length_sec: float = 2,
    max_length_sec: float = 10,
) -> List[Tuple[float, float]]:
    energy = _calculate_energy(audio_array)
    threshold = np.mean(energy)  # A simple threshold based on mean energy
    peak_times = _find_energy_peaks(energy, threshold, sample_rate)

    segments = []
    start_time = peak_times[0] if len(peak_times) > 0 else None

    for time in peak_times:
        if start_time is None:
            start_time = time
        elif time - start_time > max_length_sec:
            segments.append((start_time, start_time + max_length_sec))
            start_time = time
        elif time - start_time >= min_length_sec:
            end_time = time
            segments.append((start_time, end_time))
            start_time = None

    # Ensure the last segment is captured
    if (
        start_time is not None
        and (len(audio_array) / sample_rate) - start_time >= min_length_sec
    ):
        end_time = min(start_time + max_length_sec, len(audio_array) / sample_rate)
        segments.append((start_time, end_time))

    return segments


def energy_based_chunker(
    audio_array: np.ndarray,
    sample_rate: int,
    chunk_length_sec: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    chunks = []
    timestamps = []
    segments = _enegery_audio_segmenter(
        audio_array, sample_rate, max_length_sec=chunk_length_sec
    )
    for start_time, end_time in segments:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        chunk = audio_array[start_sample:end_sample]
        chunks.append(chunk)
        timestamps.append([start_time, end_time])

    timestamps = np.array(timestamps)
    return chunks, timestamps
