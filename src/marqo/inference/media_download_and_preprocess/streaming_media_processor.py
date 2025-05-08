"""Functions used to download and preprocess audio and video files"""

import math
import os
import subprocess
# for multimodal processing
import tempfile

import ffmpeg
from torch import Tensor

from marqo.core.exceptions import InternalError
from marqo.core.inference.api import *
from marqo.core.models.marqo_index import *
from marqo.inference.native_inference.embedding_models.languagebind_model import LanguagebindPreprocessor


class StreamingMediaProcessor:

    VIDEO_CPU_TIMOUT_OUT_MULTIPLIER = 10
    AUDIO_CPU_TIMOUT_OUT_MULTIPLIER = 10
    VIDEO_GPU_TIMOUT_OUT_MULTIPLIER = 10

    def __init__(
            self,
            url: str,
            preprocessors: LanguagebindPreprocessor,
            preprocessing_config: Union[AudioPreprocessingConfig, VideoPreprocessingConfig],
            enable_video_gpu_acceleration: bool = False
    ):
        """
        Instantiate the StreamingMediaProcessor class.

        Args:
            url: The URL of the media file to be processed.
            preprocessors: The LanguagebindPreprocessor instance to be used for preprocessing the media.
            preprocessing_config: The configuration for preprocessing the media, which includes the modality (audio or video),
            enable_video_gpu_acceleration: Whether to enable GPU acceleration for video processing.
        Raises:
            MediaExceedsMaxSizeError: If the media file size exceeds the maximum allowed size.
            MediaDownloadError: If there is an error downloading the media file.
        """
        self.url = url
        self.modality = preprocessing_config.modality
        
        self.media_download_header = self._convert_headers_to_cli_format(preprocessing_config.download_header)
        self.total_size, self.duration, self.probed_modality = self._fetch_file_metadata()

        if self.modality != self.probed_modality:
            raise MediaMismatchError(
                f"Error processing media file {self.url}. The provided modality {self.modality} does not match the "
                f"detected modality {self.probed_modality}. Please check your media file and try again. If you are using "
                f"a structured index, check if your media file matches the field type"
            )

        if self.total_size > preprocessing_config.max_media_size_bytes:
            raise MediaExceedsMaxSizeError(
                f"File size ({self.total_size / 1024 / 1024:.2f} MB) "
                f"exceeds the maximum allowed size of {preprocessing_config.max_media_size_bytes / 1024 / 1024:.2f} MB"
            )

        if preprocessing_config.should_chunk:
            self.split_length = preprocessing_config.chunk_config.split_length
            self.split_overlap = preprocessing_config.chunk_config.split_overlap
        else:
            self.split_length = self.duration
            self.split_overlap = 0
        
        self.preprocessors = preprocessors
        
        self.enable_video_gpu_acceleration = enable_video_gpu_acceleration

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

    def _infer_modality_from_probe(self, modality_list: list[str], format_name: Optional[str]) -> Optional[Modality]:
        """
        Infer the modality from the probed media file. This is used to determine whether the media is audio or video.
        """
        if Modality.VIDEO in modality_list:
            # Images are also considered as video in ffmpeg, so we need to check the format name to
            # differentiate between video and image
            if "image" in format_name or "_pipe" in format_name:
                return Modality.IMAGE
            else:
                return Modality.VIDEO
        elif Modality.AUDIO in modality_list:
            return Modality.AUDIO
        else:
            return None

    def _fetch_file_metadata(self) -> Tuple[float, float, Optional[Modality]]:
        """
        Fetch the metadata of the media file using ffmpeg. This includes the size, duration, and modality of the
        media file.

        Returns:
            Tuple[float, float, str]: A tuple containing the size (in bytes), duration (in seconds), and modality of the
            media file.

        """
        try:
            probe_options = {
                'v': 'error',
                'show_entries': 'stream=codec_type,format=size,duration,format_name',
                'of': 'json',
                'probesize': '256K',  # Probe only the first 256KB
            }

            if self.media_download_header:
                probe_options['headers'] = self.media_download_header

            probe = ffmpeg.probe(self.url, **probe_options)

            size = int(probe['format'].get('size', 0))
            duration = float(probe['format'].get('duration', 0))
            format_name = probe['format'].get('format_name', "")
            modality_list = [codec_type.get('codec_type', "") for codec_type in probe['streams']]
            modality = self._infer_modality_from_probe(modality_list, format_name)

            return size, duration, modality

        except ffmpeg.Error as e:
            raise MediaDownloadError(f"Error fetching metadata: {e.stderr.decode()}") from e

    def _get_output_file_path(self, temp_dir, chunk_start):
        extension = 'mp4' if self.modality == Modality.VIDEO else 'wav'
        return os.path.join(temp_dir, f"chunk_{chunk_start}.{extension}")

    def process_media(self) -> list[Tuple[str, Tensor]]:
        """
        Process the media file by splitting it into chunks and downloading each chunk, and apply the languagebind
        preprocessor to each chunk.

        Returns:
            list[Tuple[str, Tensor]]: A list of tuples, where each tuple contains the chunk content and the

        Raise:
            MediaDownloadError: If there is an error downloading or processing the media file.
        """
        processed_chunks: list[Tuple[str, Tensor]] = []
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

                # We expect no error in the preprocessing step
                processed_chunk_tensor: Tensor = self.preprocessors.preprocess(
                    [output_file], modality=self.modality)[0]

                processed_chunks.append(
                    (f"[{chunk_start:.1f}, {chunk_end:.1f}]", processed_chunk_tensor)
                )
        if not processed_chunks:
            raise MediaDownloadError(
                f"Error processing media file {self.url}: No chunks were successfully processed"
            )
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

        if self.media_download_header:
            # -headers must appear before -i
            ffmpeg_command.extend(['-headers', self.media_download_header])

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
        if self.media_download_header:
            # -headers must appear before -i
            ffmpeg_command.extend(['-headers', self.media_download_header])

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