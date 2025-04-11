"""Functions used to download and preprocess audio and video files"""

import math
import os
import subprocess
# for multimodal processing
import tempfile
from typing import Tuple

import ffmpeg
import torch

from marqo.core.exceptions import InternalError
from marqo.core.models.marqo_index import *
from marqo.s2_inference.errors import MediaDownloadError
from marqo.core.inference.api.modality import Modality
from marqo.tensor_search.models.preprocessors_model import Preprocessors


class StreamingMediaProcessor:

    VIDEO_CPU_TIMOUT_OUT_MULTIPLIER = 10
    AUDIO_CPU_TIMOUT_OUT_MULTIPLIER = 10
    VIDEO_GPU_TIMOUT_OUT_MULTIPLIER = 10

    def __init__(
            self, url: str, device: str, modality: Modality,
            preprocessors: Preprocessors, audio_preprocessing: AudioPreProcessing = None,
            video_preprocessing: VideoPreProcessing = None, media_download_headers: Optional[Dict[str, str]] = None,
            enable_video_gpu_acceleration: bool = False
    ):
        self.url = url
        self.device = device
        self.modality = modality
        self.audio_preprocessing = audio_preprocessing
        self.video_preprocessing = video_preprocessing
        self.preprocessor = preprocessors.get_preprocessor(modality)
        self.media_download_headers = self._convert_headers_to_cli_format(media_download_headers)
        self.total_size, self.duration = self._fetch_file_metadata()
        self.enable_video_gpu_acceleration = enable_video_gpu_acceleration
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

    def _convert_headers_to_cli_format(self, raw_media_download_headers: Optional[Dict] = None) -> str:
        """
        A helper function to convert the media download headers into a format that can be passed to ffmpeg in
        subprocess calls.

        Examples:
            If the headers are {"key1": "value1", "key2": "value2"}, the function will return a string
            "key1: value1\r\nkey2: value2"

        Returns:
            str: The headers in the required format. An empty string if no headers or None are provided.
        """
        if raw_media_download_headers is None or raw_media_download_headers == {}:
            return ""
        elif not isinstance(raw_media_download_headers, dict):
            raise InternalError("media_download_headers should be a dictionary")
        return "\r\n".join([f"{key}: {value}" for key, value in raw_media_download_headers.items()])

    def _fetch_file_metadata(self) -> Tuple[float, float]:
        try:
            probe_options = {
                'v': 'error',
                'show_entries': 'format=size,duration',
                'of': 'json',
                'probesize': '256K',  # Probe only the first 256KB
            }

            if self.media_download_headers:
                probe_options['headers'] = self.media_download_headers

            probe = ffmpeg.probe(self.url, **probe_options)

            size = int(probe['format'].get('size', 0))
            duration = float(probe['format'].get('duration', 0))

            return size, duration

        except ffmpeg.Error as e:
            raise MediaDownloadError(f"Error fetching metadata: {e.stderr.decode()}") from e

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
                    chunk_end = min([chunk_start + chunk_duration, self.duration])

                output_file = self._get_output_file_path(temp_dir, chunk_start)

                try:
                    if self.modality == Modality.VIDEO:
                        output_file = self.fetch_video_chunk(
                            start_time=chunk_start,
                            duration=chunk_end - chunk_start,
                            output_file=output_file,
                        )
                    elif self.modality == Modality.AUDIO:  # AUDIO
                        output_file = self.fetch_audio_chunk(
                            start_time=chunk_start,
                            duration=chunk_end - chunk_start,
                            output_file=output_file,
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

    def fetch_video_chunk(self, start_time: float, duration: float, output_file: str) -> str:
        """
        Fetch a video chunk from the url, starting at start_time and lasting duration seconds. Return the path to the
        downloaded video chunk.
        Args:
            start_time: The start time of the video chunk
            duration: The duration of the video chunk
            output_file: The path to save the video chunk

        Returns:
            THe path to the downloaded video chunk

        Raises:
            MediaDownloadError: If there is an error downloading the video chunk
        """
        ffmpeg_command = [
            'ffmpeg',
            '-y',  # Enable overwrite
            '-v', 'error',  # Suppress warnings and other output
        ]

        if self.media_download_headers:
            # -headers must appear before -i
            ffmpeg_command.extend(['-headers', self.media_download_headers])

        if self.enable_video_gpu_acceleration:
            ffmpeg_command.extend([
                '-ss', str(start_time),  # Start time
                '-t', str(duration),  # Duration
                '-hwaccel', 'cuda',  # Use GPU acceleration
                '-hwaccel_output_format', 'cuda',  # Use GPU acceleration
                '-i', self.url,  # Input file
                '-c:a', 'copy', # Copy audio codec to speed up the conversion process by avoiding unnecessary re-encoding of the audio stream.
                '-c:v', 'h264_nvenc', # Use NVIDIA NVENC H.264 encoder
                '-b:v', '5M', # Set the video bitrate to 5M
                output_file
            ])
            timeout = duration * self.VIDEO_GPU_TIMOUT_OUT_MULTIPLIER
        else:
            ffmpeg_command.extend([
                '-ss', str(start_time),  # Start time
                '-t', str(duration),  # Duration
                '-i', self.url,  # Input file
                '-vcodec', 'libx264',
                '-acodec', 'aac',
                '-f', 'mp4',
                output_file
            ])
            timeout = duration * self.VIDEO_CPU_TIMOUT_OUT_MULTIPLIER

        base_error_message = f"Error downloading the video chunk with url={self.url}, start_time={start_time},"

        try:
            self._run_ffmpeg_command(ffmpeg_command, timeout, base_error_message)
        except (MediaDownloadError, InternalError):
            if os.path.exists(output_file): # Remove the file if it was created
                os.remove(output_file)
            raise
        return output_file

    def fetch_audio_chunk(self, start_time: float, duration: float, output_file: str) -> str:
        """
        Fetch an audio chunk from the url, starting at start_time and lasting duration seconds. Return the path to the
        downloaded audio chunk.
        Args:
            start_time: The start time of the audio chunk
            duration: The duration of the audio chunk
            output_file: The path to save the audio chunk

        Returns:
            The path to the downloaded audio chunk
        """
        ffmpeg_command = [
            'ffmpeg',
            '-y', # Enable overwrite
            '-v', 'error',  # Suppress warnings and other output
        ]
        if self.media_download_headers:
            # -headers must appear before -i
            ffmpeg_command.extend(['-headers', self.media_download_headers])

        ffmpeg_command.extend(
            [
                '-i', str(self.url),  # Input file
                '-ss', str(start_time),  # Start time
                '-t', str(duration),  # Duration
                '-acodec', 'pcm_s16le',  # Audio codec
                '-ar', '44100',  # Audio sample rate
                '-f', 'wav',  # Output format
                output_file  # Output file
            ]
        )
        timeout = duration * self.AUDIO_CPU_TIMOUT_OUT_MULTIPLIER

        base_error_message = f"Error downloading the audio chunk with url={self.url}, start_time={start_time},"
        try:
            self._run_ffmpeg_command(ffmpeg_command, timeout, base_error_message)
        except (MediaDownloadError, InternalError):
            if os.path.exists(output_file): # Remove the file if it was created
                os.remove(output_file)
            raise
        return output_file

    def _run_ffmpeg_command(
            self, ffmpeg_command: List[str], timeout: float, base_error_message: str
    ) -> None:
        """Call ffmpeg with the given command and timeout.

        Args:
            ffmpeg_command: The ffmpeg command to run
            timeout: The maximum time to wait for the command to complete
            base_error_message: The base error message to use in case of an error
        Raises:
            MediaDownloadError: If there is an error downloading or the operation times out.
            InternalError: If there is an expected error running the ffmpeg command, such as OSError when there is
                no ffmpeg installed, or ValueError when the command is invalid.
        """
        try:
            _ = subprocess.run(
                ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                text=True, timeout=timeout
            )
        except subprocess.CalledProcessError as e:
            raise MediaDownloadError(f"{base_error_message} Original error: {e.stderr}") from e
        except subprocess.TimeoutExpired as e:
            raise MediaDownloadError(f"{base_error_message} the download operation timed out after {timeout} seconds") \
                from e
        except (OSError, ValueError) as e:
            raise InternalError(f"Error running ffmpeg command: {e}") from e