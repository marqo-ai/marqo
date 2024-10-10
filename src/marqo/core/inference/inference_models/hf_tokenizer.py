import html
from typing import Union, List

import ftfy
import regex as re
import torch


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

class HFTokenizer:
    # HuggingFace _tokenizer wrapper
    # Check https://github.com/mlfoundations/open_clip/blob/16e229c596cafaec46a4defaf27e0e30ffcca12d/src/open_clip/tokenizer.py#L188-L201
    def __init__(self, tokenizer_name: str):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, texts: Union[str, List[str]]) -> torch.Tensor:
        # same cleaning as for default _tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]
        texts = [whitespace_clean(basic_clean(text)) for text in texts]
        input_ids = self.tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True).input_ids
        return input_ids