"""Functions used to download and preprocess audio and video files"""

from contextlib import contextmanager
import torch
import math

# for multimodal processing
import tempfile
import os
import subprocess
from urllib.parse import urlparse
from typing import List, Dict
import time
import ffmpeg

from torchvision.transforms import Compose
from marqo.core.models.marqo_index import *
from marqo.s2_inference.s2_inference import  Modality


class StreamingMediaProcessor:
    def __init__(self, url: str, headers: Dict[str, str], modality: Modality, marqo_index: MarqoIndex, preprocessors: Dict[str, Compose]):
        self.url = url
        self.headers = headers
        self.modality = modality
        self.marqo_index = marqo_index
        self.preprocessors = preprocessors
        self.total_size, self.duration = self.fetch_file_metadata()
        self.preprocessor = self.preprocessors[modality]
        
        if modality == Modality.VIDEO:
            video_preprocessing = marqo_index.video_preprocessing
            print(f"video_preprocessing: {video_preprocessing}")
            if video_preprocessing is not None:
                self.split_length = marqo_index.video_preprocessing.split_length
                self.split_overlap = marqo_index.video_preprocessing.split_overlap
            else:
                self.split_length = 20
                self.split_overlap = 3

        elif modality == Modality.AUDIO:
            audio_preprocessing = marqo_index.audio_preprocessing
            print(f"audio_preprocessing: {audio_preprocessing}")
            if audio_preprocessing is not None:
                self.split_length = marqo_index.audio_preprocessing.split_length
                self.split_overlap = marqo_index.audio_preprocessing.split_overlap
            else:
                self.split_length = 20
                self.split_overlap = 3
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        
        print(f"from StreamingMediaProcessor, self.split_length: {self.split_length}")
        print(f"from StreamingMediaProcessor, self.split_overlap: {self.split_overlap}")

        self.chunk_size = None #self.estimate_chunk_size()

        print(f"from StreamingMediaProcessor, self.total_size: {self.total_size}")
        print(f"from StreamingMediaProcessor, self.duration: {self.duration}")
        print(f"from StreamingMediaProcessor, self.chunk_size: {self.chunk_size}")

    @contextmanager
    def _temp_file(self, suffix):
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                yield temp_file.name
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    def fetch_file_metadata(self):
        parsed_url = urlparse(self.url)
        is_local_file = parsed_url.scheme == '' or parsed_url.scheme == 'file'

        if not is_local_file:
            http_start_time = time.time()
            try:
                size, duration = self._fetch_metadata_http()
                http_end_time = time.time()
                http_duration = http_end_time - http_start_time
                print(f"HTTP -> ffprobe method took {http_duration:.2f} seconds")
                return size, duration
            except Exception as e:
                print(f"Error fetching metadata via HTTP: {str(e)}. Falling back to ffprobe.")

        ffprobe_start_time = time.time()
        try:
            size, duration = self._fetch_metadata_ffprobe()
            ffprobe_end_time = time.time()
            ffprobe_duration = ffprobe_end_time - ffprobe_start_time
            print(f"Pure ffprobe method took {ffprobe_duration:.2f} seconds")
            return size, duration
        except ffmpeg.Error as e:
            logger.error(f"Error fetching metadata: {e.stderr.decode()}")
            raise

    def fetch_file_metadata(self):
        start_time = time.time()
        print(f"Starting fetch_file_metadata for URL: {self.url}")

        # If the above methods fail or it's a local file, fall back to ffprobe
        try:
            probe_options = {
                'v': 'error',
                'show_entries': 'format=size,duration',
                'of': 'json',
                'probesize': '256K' # Probe only the first 256KB
            }

            probe = ffmpeg.probe(self.url, **probe_options)
            
            size = int(probe['format'].get('size', 0))
            duration = float(probe['format'].get('duration', 0))

            print(f"from StreamingMediaProcessor, got duration: {duration}, size: {size} bytes")

            end_time = time.time()
            print(f"fetch_file_metadata using full ffprobe completed in {(end_time - start_time) * 1000:.2f} ms")
            return size, duration
        
        except ffmpeg.Error as e:
            logger.error(f"Error fetching metadata: {e.stderr.decode()}")
            raise

    def estimate_chunk_size(self):
        if self.duration:
            return int(self.total_size / self.duration * self.split_length)
        else:
            # If we couldn't estimate duration, start with a reasonable chunk size
            return 1024 * 1024  # 1 MB

    def process_media(self) -> List[Dict[str, torch.Tensor]]:
        processed_chunks = []
        chunk_duration = self.split_length
        overlap_duration = self.split_overlap
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for chunk_start in range(0, math.ceil(self.duration), chunk_duration - overlap_duration):
                chunk_end = min(chunk_start + chunk_duration, self.duration)
                
                output_file = os.path.join(temp_dir, f"chunk_{chunk_start}.{'mp4' if self.modality == Modality.VIDEO else 'wav'}")
                
                try:
                    # Use ffmpeg-python to process the chunk
                    stream = ffmpeg.input(self.url, ss=chunk_start, t=chunk_end - chunk_start)
                    
                    if self.modality == Modality.VIDEO:
                        stream = ffmpeg.output(stream, output_file, vcodec='libx264', acodec='aac', **{'f': 'mp4'})
                    else:  # AUDIO
                        stream = ffmpeg.output(stream, output_file, acodec='pcm_s16le', ar=44100, **{'f': 'wav'})
                    
                    ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                    
                    
                    if self.modality == Modality.VIDEO:
                        processed_chunk_tensor = self.preprocessor(output_file, return_tensors='pt')
                    else:  # AUDIO
                        processed_chunk_tensor = self.preprocessor(output_file, return_tensors='pt')
                    #print(f"processed_chunk_tensor: {processed_chunk_tensor}")
                    print(f"len(processed_chunk_tensor['pixel_values'].shape): {processed_chunk_tensor['pixel_values'].shape}")
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

                # Move to the next chunk start, but don't not beyond the file duration
                if chunk_end == self.duration:
                    break
                chunk_start = min(chunk_start + (chunk_duration - overlap_duration), self.duration - overlap_duration)
                
        return processed_chunks

    def _progress(self, download_total, downloaded, upload_total, uploaded):
        if download_total > 0:
            progress = downloaded / download_total * 100
            print(f"Download progress: {progress:.2f}%")

    def process_chunk(self, chunk_data: bytes, chunk_number: int) -> Optional[Dict[str, torch.Tensor]]:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4' if self.modality == Modality.VIDEO else '.wav') as temp_file:
            temp_file.write(chunk_data)
            temp_file_path = temp_file.name

        try:
            if self.modality == Modality.VIDEO:
                ffmpeg_cmd = [
                    'ffmpeg', '-i', temp_file_path, '-c:v', 'libx264', '-c:a', 'aac',
                    '-movflags', '+faststart', '-y', f"{temp_file_path}.processed.mp4"
                ]
            else:  # AUDIO
                ffmpeg_cmd = [
                    'ffmpeg', '-i', temp_file_path, '-acodec', 'pcm_s16le', '-ar', '44100',
                    '-y', f"{temp_file_path}.processed.wav"
                ]

            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            processed_file_path = f"{temp_file_path}.processed.{'mp4' if self.modality == Modality.VIDEO else 'wav'}"
            
            if self.modality == Modality.VIDEO:
                processed_chunk = self.preprocessor(processed_file_path, return_tensors='pt')
            else:  # AUDIO
                processed_chunk = self.preprocessor(processed_file_path, return_tensors='pt')

            start_time = chunk_number * (self.split_length - self.split_overlap)
            end_time = start_time + self.split_length

            return processed_chunk

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_number}: {str(e)}")
            return None

        finally:
            os.unlink(temp_file_path)
            if os.path.exists(f"{temp_file_path}.processed.mp4"):
                os.unlink(f"{temp_file_path}.processed.mp4")
            if os.path.exists(f"{temp_file_path}.processed.wav"):
                os.unlink(f"{temp_file_path}.processed.wav")