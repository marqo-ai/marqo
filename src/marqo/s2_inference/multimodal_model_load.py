"""Abstractions for Multimodal Models"""

import io
from contextlib import contextmanager
from typing import Optional, Union, List

import magic
import requests

from marqo.core.inference.image_download import encode_url
from marqo.s2_inference.clip_utils import validate_url
from marqo.s2_inference.errors import MediaDownloadError
from marqo.s2_inference.types import Modality


@contextmanager
def fetch_content_sample(url, media_download_headers: Optional[dict] = None, sample_size=10240):  # 10 KB
    # It's ok to pass None to requests.get() for headers and it won't change the default headers
    """Fetch a sample of the content from the URL.

    Raises:
        HTTPError: If the response status code is not 200
    """
    response = requests.get(url, stream=True, headers=media_download_headers)
    response.raise_for_status()
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


def infer_modality(content: Union[str, List[str], bytes], media_download_headers: Optional[dict] = None) -> Modality:
    """
    Infer the modality of the content. Video, audio, image or text.
    """
    if isinstance(content, str):
        if not validate_url(content):
            return Modality.TEXT

        # Encode the URL
        encoded_url = encode_url(content)
        extension = encoded_url.split('.')[-1].lower()
        if extension in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            return Modality.IMAGE
        elif extension in ['mp4', 'avi', 'mov']:
            return Modality.VIDEO
        elif extension in ['mp3', 'wav', 'ogg']:
            return Modality.AUDIO
        if validate_url(encoded_url):
            # Use context manager to handle content sample
            try:
                with fetch_content_sample(encoded_url, media_download_headers) as sample:
                    mime = magic.from_buffer(sample.read(), mime=True)
                    if mime.startswith('image/'):
                        return Modality.IMAGE
                    elif mime.startswith('video/'):
                        return Modality.VIDEO
                    elif mime.startswith('audio/'):
                        return Modality.AUDIO
            except requests.exceptions.RequestException as e:
                raise MediaDownloadError(f"Error downloading media file {content}: {e}") from e
            except magic.MagicException as e:
                raise MediaDownloadError(f"Error determining MIME type for {encoded_url}: {e}") from e
            except IOError as e:
                raise MediaDownloadError(f"IO error while processing {encoded_url}: {e}") from e

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

# class LanguageBindEncoder(ModelEncoder):
#     def __init__(self, model: MultimodalModel):
#         self.model = model
#         self.tokenizer = self._get_tokenizer()
#
#     @contextmanager
#     def _temp_file(self, suffix):
#         temp_file = None
#         try:
#             with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
#                 yield temp_file.name
#         finally:
#             if os.path.exists(temp_file.name):
#                 os.unlink(temp_file.name)
#
#     def _get_tokenizer(self):  # this is used for text only
#         if 'image' in self.model.clip_type:
#             pretrained_ckpt = 'LanguageBind/LanguageBind_Image'
#             return LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt,
#                                                               cache_dir=f'{ModelCache.languagebind_cache_path}/tokenizer_cache_dir')
#         else:
#             first_model = next(iter(self.model.clip_type.values()))
#             pretrained_ckpt = f'LanguageBind/{first_model}'
#             if "video" in first_model.lower():
#                 return LanguageBindVideoTokenizer.from_pretrained(pretrained_ckpt,
#                                                                   cache_dir=f'{ModelCache.languagebind_cache_path}/tokenizer_cache_dir')
#             else:
#                 return LanguageBindAudioTokenizer.from_pretrained(pretrained_ckpt,
#                                                                   cache_dir=f'{ModelCache.languagebind_cache_path}/tokenizer_cache_dir')
#
#     def _normalize(self, outputs):
#         return outputs / outputs.norm(dim=-1, keepdim=True)
#
#     def preprocessor(self, modality):
#         if not hasattr(self, '_preprocessors'):
#             self._preprocessors = {}
#
#         if modality not in self._preprocessors:
#             preprocessors = {
#                 Modality.VIDEO: LanguageBindVideoProcessor,
#                 Modality.AUDIO: LanguageBindAudioProcessor,
#                 Modality.IMAGE: LanguageBindImageProcessor
#             }
#             if modality in self.model.clip_type:
#                 self._preprocessors[modality] = preprocessors[modality](self.model.model.modality_config[modality])
#
#         return self._preprocessors.get(modality)
#
#     def encode(self, content, modality, media_download_headers: Optional[Dict]=None, normalize=True, **kwargs):
#         inputs = {}
#
#         if modality == Modality.TEXT:
#             inputs[Modality.TEXT] = to_device(
#                 self.tokenizer(content, max_length=77, padding='max_length', truncation=True, return_tensors='pt'),
#                 self.model.device
#             )['input_ids']
#
#         elif modality == Modality.IMAGE:
#             with self._temp_file('.png') as temp_filename:
#                 content = content[0] if isinstance(content, list) else content
#                 if isinstance(content, Image):
#                     content.save(temp_filename, format='PNG')
#                 elif isinstance(content, bytes):
#                     with open(temp_filename, 'wb') as f:
#                         f.write(content)
#                 elif isinstance(content, str) and "http" in content:
#                     self._download_content(content, temp_filename, media_download_headers, modality)
#                 else:
#                     return self.encode([content], normalize=normalize, modality=Modality.TEXT)
#
#                 preprocessed_image = self.preprocessor(Modality.IMAGE)([temp_filename], return_tensors='pt')
#                 inputs['image'] = to_device(preprocessed_image, self.model.device)['pixel_values']
#
#         elif modality in [Modality.AUDIO, Modality.VIDEO]:
#             if isinstance(content, str) and "http" in content:
#                 suffix = ".mp4" if modality == Modality.VIDEO else ".wav"
#                 with self._temp_file(suffix) as temp_filename:
#                     self._download_content(content, temp_filename, media_download_headers, modality)
#                     preprocessed_content = self.preprocessor(modality)([temp_filename], return_tensors='pt')
#                     inputs[modality.value] = to_device(preprocessed_content, self.model.device)['pixel_values']
#
#             elif isinstance(content, list) and 'pixel_values' in content[0]:
#                 # If media has already been preprocessed
#                 inputs[modality.value] = to_device(content[0], self.model.device)['pixel_values']
#             elif isinstance(content[0], str) and 'http' in content[0]:
#                 return self.encode(content[0], modality=modality, normalize=normalize, media_download_headers=media_download_headers)
#             else:
#                 raise ValueError(f"Unsupported {modality.value} content type: {type(content)}, content: {content}")
#
#         with torch.no_grad():
#             embeddings = self.model.model(inputs)
#
#         embeddings = embeddings[modality.value]
#
#         if normalize:
#             embeddings = self._normalize(embeddings)
#
#         return embeddings.cpu().numpy()
#
#
#     def _download_content(self, url, filename, media_download_headers: Optional[Dict]=None, modality: Optional[str]=None):
#         # 3 seconds for images, 20 seconds for audio and video
#         timeout_ms = 3000 if filename.endswith(('.png', '.jpg', '.jpeg')) else 20000
#
#         buffer = download_image_from_url(url, media_download_headers, timeout_ms, modality)
#
#         with open(filename, 'wb') as f:
#             f.write(buffer.getvalue())
