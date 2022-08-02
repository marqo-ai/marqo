# from torch import FloatTensor
# from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
import onnxruntime
import onnx

from marqo.s2_inference.types import *
from marqo.s2_inference.clip_utils import (
    format_and_load_CLIP_image,
    format_and_load_CLIP_images,
    _is_image,
)
from marqo.s2_inference.sbert_onnx_utils import ModelCache
from marqo.s2_inference.logger import get_logger
logger = get_logger(__name__)

def normalize_2d(inputs):

    is_valid = False
    if isinstance(inputs, FloatTensor):
        n_dims = inputs.dim()
        if n_dims == 2:
            row_sums = inputs.norm(dim=-1, keepdim=True)
            is_valid = True
    elif isinstance(inputs, ndarray):
        n_dims = inputs.ndim
        if n_dims == 2:
            row_sums = np.linalg.norm(inputs, axis=1, ord=2)[:, np.newaxis]
            is_valid = True
    elif isinstance(inputs, list):
        return normalize_2d(np.array(inputs))
    else:
        raise TypeError(f"unrecognized type {type(inputs)}")

    if is_valid:
        return inputs / row_sums

    raise TypeError(f"expected 2D matrix for normalization but received {n_dims}")  
    


class Textual(nn.Module):
    """https://colab.research.google.com/drive/1YqwLsBEP2qn3M_tgqXIKCYEjtVfFoKMy#scrollTo=WkAB99F-pneX

    Args:
        nn (_type_): _description_
    """
    def __init__(self, model):
        super().__init__()
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.token_embedding = model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # needs .float() before .argmax(  ) to work
        x = x[torch.arange(x.shape[0]), text.float().argmax(dim=-1)] @ self.text_projection

        return x


def attention(self, x: torch.Tensor):
    # onnx doesn't like multi_head_attention_forward so this is a reimplementation
    q, k, v = (torch.einsum("tbh, oh -> tbo", x, self.attn.in_proj_weight) + self.attn.in_proj_bias).contiguous().chunk(3, dim=-1)
    tgt_len = q.shape[0]
    bsz = q.shape[1]
    num_heads = self.attn.num_heads
    head_dim = q.shape[2] // num_heads
    attn_output, attn_output_weights = F._scaled_dot_product_attention(
        q.reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1),
        k.reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1),
        v.reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1), None, 0.0
        )
    attn_output = attn_output.transpose(0, 1).contiguous().view(q.shape)
    attn_output = F.linear(attn_output, self.attn.out_proj.weight, self.attn.out_proj.bias)
    return attn_output

class ConvertCLIP:

    default_onnx_params = dict(input_names=['input'], output_names=['output'],
                      export_params=True, verbose=False, opset_version=12,
                      do_constant_folding=True,
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    def __init__(self, model_name_or_path: Optional[str] = None,
                 cache_folder: Optional[str] = None,
                 onnx_folder: Optional[str] = None,
                 vision_onnx_model_name: Optional[str] = None,
                 text_onnx_model_name: Optional[str] = None,
                 enable_overwrite: Optional[bool] = False,
                 ):

        # some paths
        self.cache_folder = cache_folder
        self.onnx_folder = onnx_folder
        self.vision_onnx_model_name = vision_onnx_model_name
        self.text_onnx_model_name = text_onnx_model_name
        self.model_name_or_path = model_name_or_path

        # should we overwrite even if another exists
        self.enable_overwrite = enable_overwrite

        # final names for the export onnx models
        self.vision_export_model_name = None
        self.text_export_model_name = None

        # have a flag to see if either exists or not
        self.vision_already_exists = None
        self.text_already_exists = None

        # dummy inputs
        self.text_dummy_input = None
        self.text_dummy_input_tokenized = None
        self.vision_dummy_input = None

        #
        self.model = None
        self.preprocess = None
        self.tokenizer = None

        logger.info("getting model paths...")
        self._get_paths()
        logger.info("checking if onnx already exists...")
        self._check_exists()

        logger.info('loading original clip...')
        self._load_original_clip()
        
        if not self.vision_already_exists or not self.text_already_exists:
            logger.info('done...')
            logger.info("getting dummy inputs for conversion...")
            self._get_dummy_inputs()
            logger.info('done...')


    def _load_original_clip(self):
        import clip
        #clip.model.ResidualAttentionBlock.attention = attention
        self.model, self.preprocess = clip.load(self.model_name_or_path, device="cpu", jit=False)
        self.tokenizer = clip.tokenize
        self.model.eval()

    def _check_exists(self):

        # construct all the save paths and cache directories
        if self.vision_export_model_name is None or self.text_export_model_name is None:
            self._get_paths()

        # if doesn't exist or overwrite, then convert
        if not os.path.isfile(self.vision_export_model_name) or self.enable_overwrite:
            self.vision_already_exists = False
        else:
            self.vision_already_exists = True

        # if doesn't exist or overwrite, then convert
        if not os.path.isfile(self.text_export_model_name) or self.enable_overwrite:
            self.text_already_exists = False
        else:
            self.text_already_exists = True


    def _get_paths(self) -> None:
        """get the paths of the cache, onnx save path and output model path
        """
        if self.onnx_folder is None:
            self.onnx_folder = ModelCache.onnx_cache_path
            Path(self.onnx_folder).mkdir(parents=True, exist_ok=True)

        if self.cache_folder is None:
            self.cache_folder = ModelCache.torch_cache_path

        if self.vision_onnx_model_name is None:
            self.vision_onnx_model_name = f"{os.path.basename(self.model_name_or_path.replace('/', '_'))}_vision.onnx"

        if self.text_onnx_model_name is None:
            self.text_onnx_model_name = f"{os.path.basename(self.model_name_or_path.replace('/', '_'))}_text.onnx"

        self.vision_export_model_name = os.path.join(self.onnx_folder, f"{self.vision_onnx_model_name}") 
        self.text_export_model_name = os.path.join(self.onnx_folder, f"{self.text_onnx_model_name}") 

    def _get_dummy_inputs(self):

        if self.text_dummy_input is None:
            self.text_dummy_input = ['bingo', 'bango', 'dingo']
            self.text_dummy_input_tokenized = self.tokenizer(self.text_dummy_input).cpu()

        if self.vision_dummy_input is None:
            self.vision_dummy_input = [Image.new("RGB", (600,600), (255,255,255)), Image.new("RGB", (600,600), (255,255,255))]
            self.vision_dummy_input_processed = torch.stack([self.preprocess(_i).cpu() for _i in self.vision_dummy_input])

    def convert(self):

        if not self.text_already_exists:
            self.convert_text()
            self.text_already_exists = True
        if not self.vision_already_exists:
            self.convert_vision()
            self.vision_already_exists = True

    def convert_vision(self):

        if self.vision_dummy_input is None:
            self._get_dummy_inputs()

        visual = self.model.visual
        self.torch_export(visual, self.vision_dummy_input_processed,
                            self.vision_export_model_name, self.default_onnx_params)

        self.onnx_checker(self.vision_export_model_name)

    def convert_text(self):
        textual = Textual(self.model)

        if self.text_dummy_input is None:
            self._get_dummy_inputs()
        self.torch_export(textual, self.text_dummy_input_tokenized, self.text_export_model_name,
                          export_params=self.default_onnx_params)
        self.onnx_checker(self.text_export_model_name)

    @staticmethod
    def torch_export(model, dummy_input, path: str, export_params: dict):
        torch.onnx.export(model, dummy_input, path, **export_params)

    @staticmethod
    def onnx_checker(path: str):
        model = onnx.load(path)
        onnx.checker.check_model(model)
        del model

    def clean_up(self):
        if self.model is not None:
            del self.model
        
class CLIP_ONNX(ConvertCLIP):
    
    """
    conveniance class wrapper to make clip work easily for both text and image encoding
    """

    def __init__(self, model_type: str = "ViT-B/32", device: str = 'cpu',  
                            embedding_dim: int = None, 
                            cache_folder: Optional[str] = None,
                            onnx_folder: Optional[str] = None,
                            vision_onnx_model_name: Optional[str] = None,
                            text_onnx_model_name: Optional[str] = None,
                            enable_overwrite: Optional[bool] = False,
                            **kwargs) -> None:
        super().__init__(model_type,
                 cache_folder,
                 onnx_folder,
                 vision_onnx_model_name,
                 text_onnx_model_name,
                 enable_overwrite)

        self.convert()
        self.clean_up()

        self.model_type = model_type
        self.device = device
        self.model = None
        self.embedding_dimension = embedding_dim

    def load(self):
        self._get_onnx_provider()
        self._load_clip_session()

    def _get_onnx_provider(self) -> None:
        """determine where the model should run based on specified device
        """
        self.onnxproviders = onnxruntime.get_available_providers()
        logger.info(f"device:{self.device}")
        if self.device == 'cpu':
            self.fast_onnxprovider = 'CPUExecutionProvider'
        else:
            if 'CUDAExecutionProvider' not in self.onnxproviders:
                self.fast_onnxprovider = 'CPUExecutionProvider'
            else:
                self.fast_onnxprovider = 'CUDAExecutionProvider'

        logger.info(f"onnx_provider:{self.fast_onnxprovider}")

    def _load_clip_session(self):
        self.vision_session = onnxruntime.InferenceSession(self.vision_export_model_name,
                                                               providers=[self.fast_onnxprovider])
        self.text_session = onnxruntime.InferenceSession(self.text_export_model_name,
                                                                providers=[self.fast_onnxprovider])

    def _encode(self, input, modality):

        if modality == 'text':
            session = self.text_session        
        elif modality == 'vision':
            session = self.vision_session
        else:
            raise ValueError(f"expected modailty to be 'text' or 'vision' but received {modality}")

        onnx_input = {session.get_inputs()[0].name: input}
        output, = session.run(None, onnx_input)
        return output

    def encode_image(self, images, normalize = True):

        # default to batch encoding
        if isinstance(images, list):
            image_input = format_and_load_CLIP_images(images)
        else:
            image_input = [format_and_load_CLIP_image(images)]

        # image_input_processed = {self.vision_session.get_inputs()[0].name:self.preprocess(_img).to(self.device)} for _img in image_input]
        self.image_input_processed = np.array([self.preprocess(_img).cpu().numpy() for _img in image_input])

        if normalize:
            return normalize_2d(self._encode(self.image_input_processed, 'vision'))
        else:
            return self._encode(self.image_input_processed, 'vision')

    def encode_text(self, text, normalize = True):
        tokenized_text = self.tokenizer(text).detach().cpu().numpy().astype(np.int32)
        if normalize:
            return normalize_2d(self._encode(tokenized_text, 'text'))
        else:
            return self._encode(tokenized_text, 'text')

    def encode(self, inputs: Union[str, ImageType, List[Union[str, ImageType]]], 
                default: str = 'text', normalize = True, **kwargs) -> FloatTensor:

        infer = kwargs.pop('infer', True)

        if infer and _is_image(inputs):
            is_image = True
        else:
            is_image = False
            if default == 'text':
                is_image = False
            elif default == 'image':
                is_image = True
            else:
                raise ValueError(f"expected default='image' or default='text' but received {default}")

        if is_image:
            print("encoding image")
            return self.encode_image(inputs, normalize=normalize)
        else:
            print("encoding text")
            return self.encode_text(inputs, normalize=normalize)
