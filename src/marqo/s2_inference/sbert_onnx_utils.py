from statistics import mode
# from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
# from torch import FloatTensor

import torch
import os
import onnxruntime
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

from marqo.s2_inference.types import *
from marqo.s2_inference.logger import get_logger
logger = get_logger(__name__)

class ModelCache:

    onnx_cache_path = os.environ.get('ONNX_SAVE_PATH', './cache/models_onnx/')
    torch_cache_path = os.getenv('SENTENCE_TRANSFORMERS_HOME', './cache/models/')

class BaseTransformerModels:

    names = ('albert-base-v1', 'albert-base-v2', 'albert-large-v1', 'albert-large-v2', 'albert-xlarge-v1', 'albert-xlarge-v2', 'albert-xxlarge-v1', 'albert-xxlarge-v2', 'bert-base-cased-finetuned-mrpc', 'bert-base-cased', 'bert-base-chinese', 'bert-base-german-cased', 'bert-base-german-dbmdz-cased', 'bert-base-german-dbmdz-uncased', 'bert-base-multilingual-cased', 'bert-base-multilingual-uncased', 'bert-base-uncased', 'bert-large-cased-whole-word-masking-finetuned-squad', 'bert-large-cased-whole-word-masking', 'bert-large-cased', 'bert-large-uncased-whole-word-masking-finetuned-squad', 'bert-large-uncased-whole-word-masking', 'bert-large-uncased', 'camembert-base', 'ctrl', 'distilbert-base-cased-distilled-squad', 'distilbert-base-cased', 'distilbert-base-german-cased', 'distilbert-base-multilingual-cased', 'distilbert-base-uncased-distilled-squad',
                                    'distilbert-base-uncased-finetuned-sst-2-english', 'distilbert-base-uncased', 'distilgpt2', 'distilroberta-base', 'gpt2-large', 'gpt2-medium', 'gpt2-xl', 'gpt2', 'openai-gpt', 'roberta-base-openai-detector', 'roberta-base', 'roberta-large-mnli', 'roberta-large-openai-detector', 'roberta-large', 't5-11b', 't5-3b', 't5-base', 't5-large', 't5-small', 'transfo-xl-wt103', 'xlm-clm-ende-1024', 'xlm-clm-enfr-1024', 'xlm-mlm-100-1280', 'xlm-mlm-17-1280', 'xlm-mlm-en-2048', 'xlm-mlm-ende-1024', 'xlm-mlm-enfr-1024', 'xlm-mlm-enro-1024', 'xlm-mlm-tlm-xnli15-1024', 'xlm-mlm-xnli15-1024', 'xlm-roberta-base', 'xlm-roberta-large-finetuned-conll02-dutch', 'xlm-roberta-large-finetuned-conll02-spanish', 'xlm-roberta-large-finetuned-conll03-english', 'xlm-roberta-large-finetuned-conll03-german', 'xlm-roberta-large', 'xlnet-base-cased', 'xlnet-large-cased')

class Invalid:

    paths = (None, '', ' ', "", " ", "''", '""')

class Ignore:

    files = ('flax_model.msgpack', 'rust_model.ot', 'tf_model.h5')



class SBERT_ONNX(object):

    """takes and existing huggingface or sbert model and converts to onnx and loads ready for inference on cpu or gpu
    TODO: split the convert and loading/encoding
    """

    def __init__(self, model_name_or_path: Optional[str] = None,
                 device: Optional[str] = None,
                 cache_folder: Optional[str] = None,
                 onnx_folder: Optional[str] = None,
                 onnx_model_name: Optional[str] = None,
                 embedding_dim: Optional[str] = None,
                 enable_overwrite: Optional[bool] = False,
                 max_seq_length: int = 128,
                 lower_case: bool = True,
                 ):
       

        self.device = device
        self.cache_folder = cache_folder
        self.onnx_folder = onnx_folder
        self.onnx_model_name = onnx_model_name
        self.enable_overwrite = enable_overwrite
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.do_lower_case = lower_case
        self.embedding_dim = embedding_dim

        self.fast_onnxprovider = None
        self.onnxproviders = None
        self.model_path = None
        self.export_model_name = None
        self.model = None
        self.tokenizer = None
        self.session = None

        self._get_paths()
        self._get_onnx_provider()

    def load(self) -> None:
        """this does all the steps to get the onnx model
        """
        self._prepare()
        self._convert_to_onnx()
        self._load_sbert_session()
        logger.info(f"loaded {self.onnx_model_name} succesfully")

    def _get_paths(self) -> None:
        """get the paths of the cache, onnx save path and output model path
        """
        if self.onnx_folder is None:
            self.onnx_folder = ModelCache.onnx_cache_path
            Path(self.onnx_folder).mkdir(parents=True, exist_ok=True)

        if self.cache_folder is None:
            self.cache_folder = ModelCache.torch_cache_path

        if self.onnx_model_name is None:
            self.onnx_model_name = f"{os.path.basename(self.model_name_or_path.replace('/', '_'))}.onnx"
    
        self.export_model_name = os.path.join(self.onnx_folder, f"{self.onnx_model_name}") 

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

    def _prepare(self) -> None:
        """load the model and put it in eval mode
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, do_lower_case=self.do_lower_case)
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path)

        self.model.eval()

    def _load_sbert_session(self) -> None:

        """
        create the onnx session for running the model
        also see 
        https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers
        https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/bert_perf_test.py
        """

        sess_options = onnxruntime.SessionOptions()
        # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(
            self.export_model_name, sess_options, providers=[self.fast_onnxprovider])

    def _convert_to_onnx(self) -> None:
        """converts from pytorch to onnx
        """
        st = ['hello, how are you']
        inputs = self.tokenizer(
            st,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt")

        if self.enable_overwrite or not os.path.exists(self.export_model_name):
            with torch.no_grad():
                symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
                torch.onnx.export(self.model,                                            # model being run
                                  # model input (or a tuple for multiple inputs)
                                  args=tuple(inputs.values()),
                                  # where to save the model (can be a file or file-like object)
                                  f=self.export_model_name,
                                  # the ONNX version to export the model to
                                  opset_version=11,
                                  # whether to execute constant folding for optimization
                                  do_constant_folding=True,
                                  input_names=['input_ids',                         # the model's input names
                                               'attention_mask',
                                               'token_type_ids'],
                                  # the model's output names
                                  output_names=['start', 'end'],
                                  dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                                'attention_mask': symbolic_names,
                                                'token_type_ids': symbolic_names,
                                                'start': symbolic_names,
                                                'end': symbolic_names})
            logger.info(f"Model exported at: {self.export_model_name}")

            # from onnxruntime.transformers import optimizer
            # optimized_model = optimizer.optimize_model(self.export_model_name, model_type='bert', num_heads=6, hidden_size=768//2)
            # optimized_model.convert_float_to_float16()
            # optimized_model.save_model_to_file(self.export_model_name)

    @staticmethod
    def normalize(outputs: FloatTensor) -> FloatTensor:
        """normalizes vector or matrix to have unit length across rows

        Args:
            outputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        return outputs/outputs.norm(dim=-1, keepdim=True)

    @staticmethod
    def mean_pooling(token_embeddings: FloatTensor, attention_mask: FloatTensor) -> FloatTensor:
        """performs mean average pooling over a matrix with a supplied attention mask

        Args:
            token_embeddings (FloatTensor): _description_
            attention_mask (FloatTensor): _description_

        Returns:
            FloatTensor: _description_
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, sentences: Union[str, List[str]], normalize: bool = True, **kwargs) -> FloatTensor:
        """performs the vectorizxation of a sentence or list of sentences

        Args:
            sentences (Union[str, List[str]]): _description_
            normalize (bool, optional): _description_. Defaults to Normalization.normalize.

        Returns:
            FloatTensor: _description_
        """
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]

        # maybbe change to np and remove the .cpu().numpy calss?
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}

        ort_outputs = self.session.run(None, ort_inputs)
        result = self.mean_pooling(token_embeddings=torch.FloatTensor(ort_outputs[0]),
                                                        attention_mask=inputs.get('attention_mask'))
        if normalize:
            return self.normalize(result)

        return result


    
        
