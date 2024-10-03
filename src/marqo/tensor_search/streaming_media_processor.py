"""Functions used to download and preprocess audio and video files"""

import math
import os
import subprocess
# for multimodal processing
import tempfile
import time
from typing import List, Dict, Union, Any

import ffmpeg
import torch

from marqo.core.models.marqo_index import *
from marqo.s2_inference.errors import MediaDownloadError
from marqo.s2_inference.multimodal_model_load import Modality
from marqo.tensor_search.models.preprocessors_model import Preprocessors
from marqo.tensor_search import index_meta_cache, utils
from marqo.tensor_search.enums import EnvVars


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
        self.enable_video_gpu_acceleration = (
                utils.read_env_vars_and_defaults(EnvVars.MARQO_ENABLE_VIDEO_GPU_ACCELERATION) == 'TRUE'
        )

        self._set_split_parameters(modality)

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
        processed_chunks: List[Dict[str, torch.Tensor]] = []
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
                    if self.modality == Modality.VIDEO:
                        output_file = self.fetch_video_chunk(
                            url=self.url,
                            start_time=chunk_start,
                            duration=chunk_end - chunk_start,
                            output_file=output_file,
                            enable_gpu_acceleration=self.enable_video_gpu_acceleration
                        )
                    elif self.modality == Modality.AUDIO:  # AUDIO
                        output_file = self.fetch_audio_chunk(
                            url=self.url,
                            start_time=chunk_start,
                            duration=chunk_end - chunk_start,
                            output_file=output_file
                        )
                    else:
                        raise ValueError(f"Unsupported modality: {self.modality}")
                except (subprocess.CalledProcessError, MediaDownloadError) as e:
                    logger.error(f"Error processing chunk starting at {chunk_start}: {e}")
                    continue  # Skip this chunk and continue with the next one

                processed_chunk_tensor = self.preprocessor(output_file, return_tensors='pt')
                processed_chunk_tensor['pixel_values'] = processed_chunk_tensor['pixel_values'].to(self.device)

                processed_chunk = {
                    'tensor': processed_chunk_tensor,
                    'start_time': chunk_start,
                    'end_time': chunk_end
                }

                processed_chunks.append(processed_chunk)
        return processed_chunks

    def _progress(self, download_total, downloaded, upload_total, uploaded):
        if download_total > 0:
            progress = downloaded / download_total * 100

    @staticmethod
    def fetch_video_chunk(url: str, start_time: float, duration: float, output_file: str,
                          enable_gpu_acceleration: bool = False) -> str:
        """
        Fetch a video chunk from the url, starting at start_time and lasting duration seconds. Return the path to the
        downloaded video chunk.
        Args:
            url: The url of the video
            start_time: The start time of the video chunk
            duration: The duration of the video chunk
            output_file: The path to save the video chunk
            enable_gpu_acceleration: Whether to use GPU acceleration for downloading the video chunk

        Returns:
            THe path to the downloaded video chunk

        Raises:
            MediaDownloadError: If there is an error downloading the video chunk
        """
        if enable_gpu_acceleration:
            ffmpeg_command = [
                'ffmpeg',
                '-y',  # Enable overwrite
                '-v', 'error', # Suppress warnings and other output
                '-ss', str(start_time), # Start time
                '-t', str(duration), # Duration
                '-hwaccel', 'cuda', # Use GPU acceleration
                '-hwaccel_output_format', 'cuda', # Use GPU acceleration
                '-i', url, # Input file
                '-c:a', 'copy', # Copy audio codec to speed up the conversion process by avoiding unnecessary re-encoding of the audio stream.
                '-c:v', 'h264_nvenc', # Use NVIDIA NVENC H.264 encoder
                '-b:v', '5M', # Set the video bitrate to 5M
                output_file
            ]
        else:
            ffmpeg_command = [
                'ffmpeg','-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', url,
                '-vcodec', 'libx264',
                '-acodec', 'aac',
                '-f', 'mp4',
                output_file
            ]
        result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise MediaDownloadError(f"Error downloading the video chunk with url={url}, start_time={start_time}, "
                                     f"duration={duration}. "
                                     f"Original error message: {result.stderr.decode()}")
        return output_file

    @staticmethod
    def fetch_audio_chunk(url: str, start_time: float, duration: float, output_file: str) -> str:
        """
        Fetch an audio chunk from the url, starting at start_time and lasting duration seconds. Return the path to the
        downloaded audio chunk.
        Args:
            url: The url of the audio
            start_time: The start time of the audio chunk
            duration: The duration of the audio chunk
            output_file: The path to save the audio chunk

        Returns:
            The path to the downloaded audio chunk
        """
        ffmpeg_command = [
            'ffmpeg', '-y' # Enable overwrite
            '-v', 'error',  # Suppress warnings and other output
            '-i', url,  # Input file
            '-ss', str(start_time),  # Start time
            '-t', duration,  # Duration
            '-acodec', 'pcm_s16le',  # Audio codec
            '-ar', '44100',  # Audio sample rate
            '-f', 'wav',  # Output format
            output_file  # Output file
        ]

        result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise MediaDownloadError(f"Error downloading the audio chunk with url={url}, start_time={start_time}, "
                                     f"duration={duration}. "
                                     f"Original error message: {result.stderr.decode()}")
        return output_file