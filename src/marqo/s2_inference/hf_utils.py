import torch
from torch import nn
from transformers import (AutoModel, AutoTokenizer)
import numpy as np

from marqo.s2_inference.sbert_utils import Model
from marqo.s2_inference.types import Union, FloatTensor, List

from marqo.s2_inference.logger import get_logger
logger = get_logger(__name__)


class HF_MODEL(Model):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.max_seq_length is None:
            self.max_seq_length = 128

    def load(self) -> None:

        self.model = AutoModelForSentenceEmbedding(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


    def encode(self, sentence: Union[str, List[str]], normalize = True, **kwargs) -> Union[FloatTensor, np.ndarray]:
        
        if isinstance(sentence, str):
            sentence = [sentence]

        if self.model is None:
            self.load()

        self.model.normalize = normalize

        inputs = self.tokenizer(sentence, padding=True, truncation=True, max_length=self.max_seq_length, return_tensors="pt").to(self.device)

        with torch.no_grad():
            return self.model.forward(**inputs)

class AutoModelForSentenceEmbedding(nn.Module):

    def __init__(self, model_name, normalize = True, pooling='mean'):
        super().__init__()
        self.model_name = model_name
        self.normalize = normalize
        self.pooling = pooling

        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if self.pooling == 'mean':
            self._pool_func = self.mean_pooling
        elif self.pooling == 'cls':
            self._pool_func = self.cls_pooling
        else:
            raise TypeError(f"{pooling} not in allowed pooling types of 'mean' or 'cls' ")

    def forward(self, **kwargs):

        model_output = self.model(**kwargs)

        embeddings = self._pool_func(model_output, kwargs['attention_mask'])

        if self.normalize:
            return nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def mean_pooling(self, model_output, attention_mask):

        token_embeddings = model_output[0]
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def cls_pooling(self, model_output, attention_mask):
        return model_output[0][:,0]