"""Functions used to download and preprocess audio and video files"""

from contextlib import contextmanager
import torch
import math

# for multimodal processing
import tempfile
import os
import subprocess
from typing import List, Dict
import time
import ffmpeg

from marqo.core.models.marqo_index import *
from marqo.s2_inference.multimodal_model_load import Modality
from marqo.tensor_search.models.preprocessors_model import Preprocessors


class StreamingMediaProcessor:
    def __init__(self, url: str, device: str, headers: Dict[str, str], modality: Modality, marqo_index_type: IndexType,
                 marqo_index_model: Model, preprocessors: Preprocessors, audio_preprocessing: AudioPreProcessing = None,
                 video_preprocessing: VideoPreProcessing = None):
        self.url = url
        self.device = device
        self.headers = headers
        self.modality = modality
        self.marqo_index_type = marqo_index_type
        self.marqo_index_model = marqo_index_model
        self.audio_preprocessing = audio_preprocessing
        self.video_preprocessing = video_preprocessing
        self.preprocessors = preprocessors
        self.preprocessor = self.preprocessors[modality]
        self.total_size, self.duration = self._fetch_file_metadata()

        self._set_split_parameters(modality)
        self._log_initialization_details()

    def _set_split_parameters(self, modality):
        preprocessing = self.video_preprocessing if modality == Modality.VIDEO else self.audio_preprocessing

        if preprocessing is not None:
            self.split_length = preprocessing.split_length
            self.split_overlap = preprocessing.split_overlap
        else:
            self.split_length = 20
            self.split_overlap = 3

        if modality not in [Modality.VIDEO, Modality.AUDIO]:
            raise ValueError(f"Unsupported modality: {modality}")

    def _log_initialization_details(self):
        # print(f"from StreamingMediaProcessor, self.split_length: {self.split_length}")
        # print(f"from StreamingMediaProcessor, self.split_overlap: {self.split_overlap}")
        # print(f"from StreamingMediaProcessor, self.total_size: {self.total_size}")
        # print(f"from StreamingMediaProcessor, self.duration: {self.duration}")
        pass

    def _fetch_file_metadata(self):
        start_time = time.time()

        try:
            probe_options = {
                'v': 'error',
                'show_entries': 'format=size,duration',
                'of': 'json',
                'probesize': '256K'  # Probe only the first 256KB
            }

            probe = ffmpeg.probe(self.url, **probe_options)

            size = int(probe['format'].get('size', 0))
            duration = float(probe['format'].get('duration', 0))

            end_time = time.time()
            return size, duration

        except ffmpeg.Error as e:
            logger.error(f"Error fetching metadata: {e.stderr.decode()}")
            raise

    def _get_output_file_path(self, temp_dir, chunk_start):
        extension = 'mp4' if self.modality == Modality.VIDEO else 'wav'
        return os.path.join(temp_dir, f"chunk_{chunk_start}.{extension}")

    def process_media(self) -> List[Dict[str, torch.Tensor]]:
        processed_chunks = []
        chunk_duration = self.split_length
        overlap_duration = self.split_overlap

        with tempfile.TemporaryDirectory() as temp_dir:
            # Calculate total number of chunks
            total_chunks = math.ceil((self.duration - overlap_duration) / (chunk_duration - overlap_duration))

            for i in range(total_chunks):
                # For the last chunk, ensure it captures the end of the media
                if i == total_chunks - 1:
                    chunk_start = max(self.duration - chunk_duration, 0)
                    chunk_end = self.duration
                else:
                    chunk_start = i * (chunk_duration - overlap_duration)
                    chunk_end = min(chunk_start + chunk_duration, self.duration)

                output_file = self._get_output_file_path(temp_dir, chunk_start)

                try:
                    # Use ffmpeg-python to process the chunk
                    stream = ffmpeg.input(self.url, ss=chunk_start, t=chunk_end - chunk_start)

                    if self.modality == Modality.VIDEO:
                        stream = ffmpeg.output(stream, output_file, vcodec='libx264', acodec='aac', **{'f': 'mp4'})
                    else:  # AUDIO
                        stream = ffmpeg.output(stream, output_file, acodec='pcm_s16le', ar=44100, **{'f': 'wav'})

                    ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

                    processed_chunk_tensor = self.preprocessor(output_file, return_tensors='pt')
                    processed_chunk_tensor['pixel_values'] = processed_chunk_tensor['pixel_values'].to(self.device)

                    processed_chunk = {
                        'tensor': processed_chunk_tensor,
                        'start_time': chunk_start,
                        'end_time': chunk_end
                    }

                    processed_chunks.append(processed_chunk)

                except subprocess.CalledProcessError as e:
                    logger.error(f"Error processing chunk starting at {chunk_start}: {e.stderr}")
                    continue  # Skip this chunk and continue with the next one
                finally:
                    # Clean up temporary files
                    if os.path.exists(output_file):
                        os.remove(output_file)

        return processed_chunks

    def _progress(self, download_total, downloaded, upload_total, uploaded):
        if download_total > 0:
            progress = downloaded / download_total * 100
