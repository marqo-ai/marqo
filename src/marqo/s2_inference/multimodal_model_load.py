"""Abstractions for Multimodal Models"""

import requests
from contextlib import contextmanager
import tempfile
import os
import validators
import magic
import io

from pydantic import BaseModel
from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from PIL.Image import Image
import torch

from marqo.s2_inference.multimodal_model_load import *
from marqo.s2_inference.languagebind import (
    LanguageBind,
    LanguageBindVideoProcessor, LanguageBindAudioProcessor, LanguageBindImageProcessor,
    to_device
)
from marqo.s2_inference.clip_utils import download_image_from_url, validate_url
from marqo.s2_inference.languagebind.image.tokenization_image import LanguageBindImageTokenizer
from marqo.s2_inference.languagebind.video.tokenization_video import LanguageBindVideoTokenizer
from marqo.s2_inference.languagebind.audio.tokenization_audio import LanguageBindAudioTokenizer
from marqo.s2_inference.configs import ModelCache


class Modality(str, Enum):
    TEXT = "language"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class MultimodalModelProperties(BaseModel):
    name: str
    loader: str
    supported_modalities: List[Modality]
    dimensions: int
    type: str = "multimodal"
    video_chunk_length: int  # in seconds
    audio_chunk_length: int  # in seconds


class MultimodalModel:
    def __init__(self, model_name: str, model_properties: Dict[str, Any], device: str):
        self.model_name = model_name
        self.properties = MultimodalModelProperties(**model_properties)
        self.device = device
        self.model = None
        self.encoder = None

    def _load_multimodal_model(self):
        if self.properties.loader == "languagebind":
            model = self._load_languagebind_model()
            return model

        elif self.properties.loader == "imagebind":
            # Load ImageBind model
            pass
        else:
            raise ValueError(f"Unsupported loader: {self.properties.loader}")

    def _load_languagebind_model(self):
        if self.model_name == "LanguageBind/Video_V1.5_FT_Audio_FT_Image":
            self.clip_type = {
                'video': 'LanguageBind_Video_V1.5_FT',
                'audio': 'LanguageBind_Audio_FT',
                'image': 'LanguageBind_Image',
            }
        elif self.model_name == "LanguageBind/Video_V1.5_FT_Image":
            self.clip_type = {
                'video': 'LanguageBind_Video_V1.5_FT',
                'image': 'LanguageBind_Image',
            }
        elif self.model_name == "LanguageBind/Audio_FT_Image":
            self.clip_type = {
                'audio': 'LanguageBind_Audio_FT',
                'image': 'LanguageBind_Image',
            }
        elif self.model_name == "LanguageBind/Audio_FT":
            self.clip_type = {
                'audio': 'LanguageBind_Audio_FT',
            }
        elif self.model_name == "LanguageBind/Video_V1.5_FT":
            self.clip_type = {
                'video': 'LanguageBind_Video_V1.5_FT',
            }
        else:
            raise ValueError(f"Unsupported LanguageBind model: {self.model_name}")
        model = LanguageBind(clip_type=self.clip_type, cache_dir=ModelCache.languagebind_cache_path).to(self.device)
        model.eval()
        return model

    def preprocessor(self, modality):
        if self.encoder is None:
            raise ValueError("Model has not been loaded yet. Call _load_model() first.")
        return self.encoder.preprocessor(modality)

    def encode(self, content, modality, **kwargs):
        if self.encoder is None:
            raise ValueError("Model has not been loaded yet. Call _load_model() first.")
        return self.encoder.encode(content, modality, **kwargs)


class ModelEncoder(ABC):
    @abstractmethod
    def encode(self, content, modality, **kwargs):
        pass


class DefaultEncoder(ModelEncoder):
    def __init__(self, model):
        self.model = model

    def encode(self, content, modality, **kwargs):
        return self.model.encode(content)


@contextmanager
def fetch_content_sample(url, sample_size=10240):  # 10 KB
    response = requests.get(url, stream=True)
    buffer = io.BytesIO()
    try:
        for chunk in response.iter_content(chunk_size=min(sample_size, 8192)):
            buffer.write(chunk)
            if buffer.tell() >= sample_size:
                break
        buffer.seek(0)
        yield buffer
    finally:
        buffer.close()
        response.close()


def infer_modality(content: Union[str, List[str], bytes]) -> Modality:
    """
    Infer the modality of the content. Video, audio, image or text.
    """
    if isinstance(content, str):
        if not validate_url(content):
            return Modality.TEXT

        extension = content.split('.')[-1].lower()
        if extension in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            return Modality.IMAGE
        elif extension in ['mp4', 'avi', 'mov']:
            return Modality.VIDEO
        elif extension in ['mp3', 'wav', 'ogg']:
            return Modality.AUDIO

        if validate_url(content):
            # Use context manager to handle content sample
            try:
                with fetch_content_sample(content) as sample:
                    mime = magic.from_buffer(sample.read(), mime=True)
                    if mime.startswith('image/'):
                        return Modality.IMAGE
                    elif mime.startswith('video/'):
                        return Modality.VIDEO
                    elif mime.startswith('audio/'):
                        return Modality.AUDIO
            except Exception as e:
                pass

        return Modality.TEXT

    elif isinstance(content, bytes):
        # Use python-magic for byte content
        mime = magic.from_buffer(content, mime=True)
        if mime.startswith('image/'):
            return Modality.IMAGE
        elif mime.startswith('video/'):
            return Modality.VIDEO
        elif mime.startswith('audio/'):
            return Modality.AUDIO
        else:
            return Modality.TEXT

    else:
        return Modality.TEXT
        # raise ValueError(f"Unsupported content type: {type(content)}.
        # It is neither a string, list of strings, nor bytes.")


class LanguageBindEncoder(ModelEncoder):
    def __init__(self, model: MultimodalModel):
        self.model = model
        self.tokenizer = self._get_tokenizer()

    @contextmanager
    def _temp_file(self, suffix):
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                yield temp_file.name
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    def _get_tokenizer(self):  # this is used for text only
        if 'image' in self.model.clip_type:
            pretrained_ckpt = 'LanguageBind/LanguageBind_Image'
            return LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt,
                                                              cache_dir=f'{ModelCache.languagebind_cache_path}/tokenizer_cache_dir')
        else:
            first_model = next(iter(self.model.clip_type.values()))
            pretrained_ckpt = f'LanguageBind/{first_model}'
            if "video" in first_model.lower():
                return LanguageBindVideoTokenizer.from_pretrained(pretrained_ckpt,
                                                                  cache_dir=f'{ModelCache.languagebind_cache_path}/tokenizer_cache_dir')
            else:
                return LanguageBindAudioTokenizer.from_pretrained(pretrained_ckpt,
                                                                  cache_dir=f'{ModelCache.languagebind_cache_path}/tokenizer_cache_dir')

    def _normalize(self, outputs):
        return outputs / outputs.norm(dim=-1, keepdim=True)

    def preprocessor(self, modality):
        if not hasattr(self, '_preprocessors'):
            self._preprocessors = {}

        if modality not in self._preprocessors:
            preprocessors = {
                Modality.VIDEO: LanguageBindVideoProcessor,
                Modality.AUDIO: LanguageBindAudioProcessor,
                Modality.IMAGE: LanguageBindImageProcessor
            }
            if modality in self.model.clip_type:
                self._preprocessors[modality] = preprocessors[modality](self.model.model.modality_config[modality])

        return self._preprocessors.get(modality)

    def encode(self, content, modality, normalize=True, **kwargs):
        inputs = {}

        if modality == Modality.TEXT:
            inputs[Modality.TEXT] = to_device(
                self.tokenizer(content, max_length=77, padding='max_length', truncation=True, return_tensors='pt'),
                self.model.device
            )['input_ids']

        elif modality == Modality.IMAGE:
            with self._temp_file('.png') as temp_filename:
                content = content[0] if isinstance(content, list) else content
                if isinstance(content, Image):
                    content.save(temp_filename, format='PNG')
                elif isinstance(content, bytes):
                    with open(temp_filename, 'wb') as f:
                        f.write(content)
                elif isinstance(content, str) and "http" in content:
                    self._download_content(content, temp_filename)
                else:
                    return self.encode([content], modality=Modality.TEXT)

                preprocessed_image = self.preprocessor(Modality.IMAGE)([temp_filename], return_tensors='pt')
                inputs['image'] = to_device(preprocessed_image, self.model.device)['pixel_values']

        elif modality in [Modality.AUDIO, Modality.VIDEO]:
            if isinstance(content, str) and "http" in content:
                suffix = ".mp4" if modality == Modality.VIDEO else ".wav"
                with self._temp_file(suffix) as temp_filename:
                    self._download_content(content, temp_filename)
                    preprocessed_content = self.preprocessor(modality)([temp_filename], return_tensors='pt')
                    inputs[modality.value] = to_device(preprocessed_content, self.model.device)['pixel_values']

            elif isinstance(content, list) and 'pixel_values' in content[0]:
                # If media has already been preprocessed
                inputs[modality.value] = to_device(content[0], self.model.device)['pixel_values']
            elif isinstance(content[0], str) and 'http' in content[0]:
                return self.encode(content[0], modality=modality)
            else:
                raise ValueError(f"Unsupported {modality.value} content type: {type(content)}, content: {content}")

        with torch.no_grad():
            embeddings = self.model.model(inputs)

        embeddings = embeddings[modality.value]

        if normalize:
            embeddings = self._normalize(embeddings)

        return embeddings.cpu().numpy()

    def _download_content(self, url, filename):
        # 3 seconds for images, 20 seconds for audio and video
        timeout_ms = 3000 if filename.endswith(('.png', '.jpg', '.jpeg')) else 20000

        buffer = download_image_from_url(url, {}, timeout_ms)

        with open(filename, 'wb') as f:
            f.write(buffer.getvalue())
