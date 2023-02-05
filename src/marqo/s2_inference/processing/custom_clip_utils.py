import regex as re
from typing import Union, List
import torch
import ftfy
import html
import os
import urllib
from tqdm import tqdm
from src.marqo.s2_inference.configs import ModelCache
def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


class HFTokenizer:
    # HuggingFace tokenizer wrapper
    # Check https://github.com/mlfoundations/open_clip/blob/16e229c596cafaec46a4defaf27e0e30ffcca12d/src/open_clip/tokenizer.py#L188-L201
    def __init__(self, tokenizer_name:str):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, texts:Union[str, List[str]], context_length:int=77) -> torch.Tensor:
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]
        texts = [whitespace_clean(basic_clean(text)) for text in texts]
        input_ids = self.tokenizer(texts, return_tensors='pt', max_length=context_length, padding='max_length', truncation=True).input_ids
        return input_ids


def download_pretrained_from_url(
        url: str,
        cache_dir: Union[str, None] = None,
):
    '''
    This function takes a clip model checkpoint url as input, downloads the model, and returns the local
    path of the downloaded file.
    Args:
        url: a valid string of the url address.
        cache_dir: the directory to store the file
    Returns:
        download_target: the local path of the downloaded file.
    '''
    buffer_size = 8192
    if not cache_dir:
        cache_dir = os.path.expanduser(ModelCache.clip_cache_path)
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(cache_dir, filename)

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.headers.get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(buffer_size)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target
