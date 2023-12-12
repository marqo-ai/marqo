import numpy as np
import librosa
from typing import List, Union, Tuple, Literal
from marqo.s2_inference.clap_utils import load_audio_from_path


def chunk_audio(
    audio: Union[str, np.ndarray],
    method: Literal["simple", "energy", "zcr", "silence", None],
    sample_rate: int = 48000,
    chunk_length_sec: int = 10,
) -> Tuple[List[np.ndarray], list]:

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

    if method == "simple" or method == "average":
        # chunk audio with a fixed chunk length
        return simple_audio_chunker(audio_array, sample_rate, chunk_length_sec)
    elif method == "energy":
        # chunk audio based on energy peaks
        return energy_based_chunker(audio_array, sample_rate, chunk_length_sec)
    elif method == "silence":
        # remove silent sections and chunk the remaining audio
        return silence_chunker(audio_array, sample_rate, chunk_length_sec)
    elif method == "zcr":
        # chunk audio based on zero crossing rate
        return zcr_chunker(audio_array, sample_rate, chunk_length_sec)
    elif method == "mfcc":
        # chunk audio based on mfcc (Mel-Frequency Cepstral Coefficients) changes
        return mfcc_based_chunker(audio_array, sample_rate, chunk_length_sec)

    raise ValueError(f"Invalid audio chunking method: {method}")


def simple_audio_chunker(
    audio_array: np.ndarray, sample_rate: int, chunk_length_sec: int
) -> Tuple[List[np.ndarray], list]:
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

    return chunks, timestamps.tolist()


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
) -> Tuple[List[np.ndarray], list]:
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
    return chunks, timestamps.tolist()

def _find_silent_sections(audio_array: np.ndarray, sample_rate: int, silence_threshold: float = 0.01) -> List[Tuple[int, int]]:
    threshold_power = silence_threshold ** 2
    power = audio_array ** 2
    is_silent = power < threshold_power
    silent_sections = []
    current_start = None

    for i, silent in enumerate(is_silent):
        if silent and current_start is None:
            current_start = i
        elif not silent and current_start is not None:
            silent_sections.append((current_start, i))
            current_start = None

    if current_start is not None:
        silent_sections.append((current_start, len(audio_array)))

    return silent_sections

def silence_chunker(audio_array: np.ndarray, sample_rate: int, chunk_length_sec: int, silence_threshold: float = 0.01) -> Tuple[List[np.ndarray], List[List[float]]]:
    silent_sections = _find_silent_sections(audio_array, sample_rate, silence_threshold)
    chunks = []
    timestamps = []
    last_end_idx = 0

    for start_idx, end_idx in silent_sections:
        if start_idx > last_end_idx:
            non_silent_chunk, non_silent_timestamps = simple_audio_chunker(audio_array[last_end_idx:start_idx], sample_rate, chunk_length_sec)
            chunks.extend(non_silent_chunk)
            timestamps.extend([[timestamp[0] + (last_end_idx / sample_rate), timestamp[1] + (last_end_idx / sample_rate)] for timestamp in non_silent_timestamps])
        last_end_idx = end_idx

    if last_end_idx < len(audio_array):
        non_silent_chunk, non_silent_timestamps = simple_audio_chunker(audio_array[last_end_idx:], sample_rate, chunk_length_sec)
        chunks.extend(non_silent_chunk)
        timestamps.extend([[timestamp[0] + (last_end_idx / sample_rate), timestamp[1] + (last_end_idx / sample_rate)] for timestamp in non_silent_timestamps])

    return chunks, timestamps

def _calculate_zero_crossing_rate(audio_array: np.ndarray, frame_size: int = 1024) -> np.ndarray:
    frames = np.array_split(audio_array, len(audio_array) // frame_size)
    zcr = np.array([np.mean(np.diff(frame > 0)) for frame in frames])
    return zcr

def zcr_chunker(audio_array: np.ndarray, sample_rate: int, chunk_length_sec: int, frame_size: int = 1024, min_chunk_length_sec: float = 0.5) -> Tuple[List[np.ndarray], List[List[float]]]:
    zcr = _calculate_zero_crossing_rate(audio_array, frame_size)
    smoothed_zcr = np.convolve(zcr, np.ones(5)/5, mode='same')  # Example of simple smoothing

    threshold = np.percentile(smoothed_zcr, 75)  
    significant_changes = np.where(smoothed_zcr > threshold)[0] * frame_size

    chunks = []
    timestamps = []

    start_idx = 0
    for end_idx in significant_changes:
        if end_idx - start_idx >= chunk_length_sec * sample_rate:
            if (end_idx - start_idx) / sample_rate >= min_chunk_length_sec:
                chunks.append(audio_array[start_idx:end_idx])
                timestamps.append([start_idx / sample_rate, end_idx / sample_rate])
            start_idx = end_idx

    if start_idx < len(audio_array):
        chunks.append(audio_array[start_idx:])
        timestamps.append([start_idx / sample_rate, len(audio_array) / sample_rate])

    return chunks, timestamps

def _calculate_mfccs(audio_array: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> np.ndarray:
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=n_mfcc)
    return mfccs

def mfcc_based_chunker(audio_array: np.ndarray, sample_rate: int, chunk_length_sec: int) -> Tuple[List[np.ndarray], List[List[float]]]:
    mfccs = _calculate_mfccs(audio_array, sample_rate)
    delta_mfccs = np.diff(mfccs, axis=1)  # Calculate the changes between each frame

    # Identify significant changes; this threshold can be fine-tuned
    threshold = np.percentile(np.abs(delta_mfccs), 50)  
    significant_changes = np.where(np.abs(delta_mfccs).sum(axis=0) > threshold)[0]

    # Convert frame numbers to sample indices
    hop_length = 512  # This is the default in librosa mfcc calculation
    change_points = significant_changes * hop_length

    # Chunk the audio based on these change points
    chunks = []
    timestamps = []

    start_idx = 0
    for end_idx in change_points:
        if end_idx - start_idx >= chunk_length_sec * sample_rate:
            chunks.append(audio_array[start_idx:end_idx])
            timestamps.append([start_idx / sample_rate, end_idx / sample_rate])
            start_idx = end_idx

    # Handle the last chunk
    if start_idx < len(audio_array):
        chunks.append(audio_array[start_idx:])
        timestamps.append([start_idx / sample_rate, len(audio_array) / sample_rate])

    return chunks, timestamps