"""Abstractions for Multimodal Models"""

import pycurl
import requests
import subprocess
from contextlib import contextmanager
import tempfile
import os

from pydantic import BaseModel
from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from PIL.Image import Image
import torch

from marqo.s2_inference.multimodal_utils import *
from marqo.s2_inference.languagebind import (
    LanguageBind, 
    LanguageBindVideoProcessor, LanguageBindAudioProcessor, LanguageBindImageProcessor,
    to_device
)
from marqo.s2_inference.clip_utils import download_image_from_url
from marqo.s2_inference.languagebind.image.tokenization_image import LanguageBindImageTokenizer
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
    video_chunk_length: int # in seconds
    audio_chunk_length: int # in seconds
    cache_dir: str = ModelCache.languagebind_cache_path

class MultimodalModel:
    def __init__(self, model_name: str, model_properties: Dict[str, Any], device: str):
        self.model_name = model_name
        self.properties = MultimodalModelProperties(**model_properties)
        self.device = device
        self.model = None 
        self.encoder = None 
        print(f"self.device: {self.device}")

    def _load_multimodal_model(self):
        if self.properties.loader == "languagebind":
            print(f"Loading LanguageBind model: {self.model_name}")
            self.clip_type = { 
                'video': 'LanguageBind_Video_V1.5_FT',
                'audio': 'LanguageBind_Audio_FT',
                'image': 'LanguageBind_Image',
            }
            model = LanguageBind(clip_type=self.clip_type, cache_dir=self.properties.cache_dir)
            model = model.to(self.device)
            model.eval()
            print(f"successfully loaded LanguageBind model: {self.model_name}")
            return model
        elif self.properties.loader == "imagebind":
            # Load ImageBind model
            pass
        else:
            raise ValueError(f"Unsupported loader: {self.properties.loader}")
        
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

class ClipEncoder(ModelEncoder):
    def __init__(self, model):
        self.model = model

    def encode(self, content, modality, **kwargs):
        if modality == Modality.TEXT:
            return self.model.encode_text(content)
        elif modality == Modality.IMAGE:
            return self.model.encode_image(content)
        else:
            raise NotImplementedError(f"CLIP does not support encoding for modality: {modality}")


def infer_modality(content: Union[str, List[str], bytes]) -> Modality:
    if isinstance(content, str):
        extension = content.split('.')[-1].lower()
        if extension in ['jpg', 'jpeg', 'png', 'gif']:
            print(f"infer_modality, content is image")
            return Modality.IMAGE
        elif extension in ['mp4', 'avi', 'mov']:
            print(f"infer_modality, content is video")
            return Modality.VIDEO
        elif extension in ['mp3', 'wav', 'ogg']:
            print(f"infer_modality, content is audio")
            return Modality.AUDIO
        else:
            print(f"infer_modality, content is text")
            return Modality.TEXT
    elif isinstance(content, bytes):
        print(f"infer_modality, content is bytes")
        # Use python-magic to infer content type
        import magic
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
        return Modality.IMAGE # REVIEW THIS CODE


class LanguageBindEncoder(ModelEncoder):
    def __init__(self, model: MultimodalModel):
        self.model = model
        self.tokenizer = self._get_tokenizer()

    @contextmanager
    def _temp_file(self, suffix):
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                yield temp_file.name
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    def _get_tokenizer(self): # this is used for text only
        pretrained_ckpt = f'lb203/LanguageBind_Image'
        return LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir=f'{self.model.properties.cache_dir}/tokenizer_cache_dir')
    
    def preprocessor(self, modality):
        preprocessors = {
            Modality.VIDEO: LanguageBindVideoProcessor,
            Modality.AUDIO: LanguageBindAudioProcessor,
            Modality.IMAGE: LanguageBindImageProcessor
        }
        return preprocessors[modality](self.model.model.modality_config[modality])
    
    def encode(self, content, modality, **kwargs):
        inputs = {}
        
        if modality == Modality.TEXT:
            inputs[Modality.TEXT] = to_device(
                self.tokenizer(content, max_length=77, padding='max_length', truncation=True, return_tensors='pt'),
                self.model.device
            )['input_ids']

        elif modality == Modality.IMAGE:
            with self._temp_file('.png') as temp_filename:
                #if isinstance(content, list):
                #    for item in content:
                #        return self.encode(item, modality=modality)
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
                print(f"preprocessed_image: {preprocessed_image}")
                print(f"preprocessed_image['pixel_values'].shape: {preprocessed_image['pixel_values'].shape}")
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

        return embeddings[modality.value].cpu().numpy()

    def _download_content_old(self, url, filename):
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        with open(filename, 'wb') as f:
            c.setopt(c.WRITEDATA, f)
            c.perform()
        c.close()
        print(f"successfully downloaded {filename}")

    def _download_content(self, url, filename):
        # 3 seconds for images, 20 seconds for audio and video
        timeout_ms = 3000 if filename.endswith(('.png', '.jpg', '.jpeg')) else 20000  

        buffer = download_image_from_url(url, {}, timeout_ms)
        
        with open(filename, 'wb') as f:
            f.write(buffer.getvalue())
        
        print(f"Successfully downloaded {filename}")