from transformers import CLIPTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "lb203/LanguageBind-Thermal": "https://huggingface.co/lb203/LanguageBind-Thermal/resolve/main/vocab.json",
    },
    "merges_file": {
        "lb203/LanguageBind-Thermal": "https://huggingface.co/lb203/LanguageBind-Thermal/resolve/main/merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "lb203/LanguageBind-Thermal": 77,
}


PRETRAINED_INIT_CONFIGURATION = {
    "lb203/LanguageBind-Thermal": {},
}

class LanguageBindThermalTokenizer(CLIPTokenizer):
    """
    Construct a CLIP tokenizer. Based on byte-level Byte-Pair-Encoding.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `<|startoftext|>`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The end of sequence token.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
            self,
            vocab_file,
            merges_file,
            errors="replace",
            unk_token="<|endoftext|>",
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",  # hack to enable padding
            **kwargs,
    ):
        super(LanguageBindThermalTokenizer, self).__init__(
            vocab_file,
            merges_file,
            errors,
            unk_token,
            bos_token,
            eos_token,
            pad_token,  # hack to enable padding
            **kwargs,)