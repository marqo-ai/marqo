import torch
from torch import nn
from transformers import (AutoModel, AutoTokenizer)
import numpy as np

from marqo.s2_inference.sbert_utils import Model
from marqo.s2_inference.types import Union, FloatTensor, List

from marqo.s2_inference.logger import get_logger

logger = get_logger(__name__)


class HF_ENDPOINT(Model):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def load(self) -> None:
        print("loaded")
       

    def encode(self, sentence: Union[str, List[str]], normalize=True, **kwargs) -> Union[FloatTensor, np.ndarray]:
        print("encoded")

