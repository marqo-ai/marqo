import os
from marqo.tensor_search import enums

class ModelCache:
    onnx_cache_path = os.environ.get('ONNX_SAVE_PATH', f'{os.environ[enums.EnvVars.MARQO_ROOT_PATH]}/cache/models_onnx/')
    torch_cache_path = os.getenv('SENTENCE_TRANSFORMERS_HOME', f'{os.environ[enums.EnvVars.MARQO_ROOT_PATH]}/cache/models/')
    clip_cache_path = os.getenv('CLIP_SAVE_PATH', f'{os.environ[enums.EnvVars.MARQO_ROOT_PATH]}/cache/clip/')

class BaseTransformerModels:

    names = ('albert-base-v1', 'albert-base-v2', 'albert-large-v1', 'albert-large-v2', 'albert-xlarge-v1', 'albert-xlarge-v2', 'albert-xxlarge-v1', 'albert-xxlarge-v2', 'bert-base-cased-finetuned-mrpc', 'bert-base-cased', 'bert-base-chinese', 'bert-base-german-cased', 'bert-base-german-dbmdz-cased', 'bert-base-german-dbmdz-uncased', 'bert-base-multilingual-cased', 'bert-base-multilingual-uncased', 'bert-base-uncased', 'bert-large-cased-whole-word-masking-finetuned-squad', 'bert-large-cased-whole-word-masking', 'bert-large-cased', 'bert-large-uncased-whole-word-masking-finetuned-squad', 'bert-large-uncased-whole-word-masking', 'bert-large-uncased', 'camembert-base', 'ctrl', 'distilbert-base-cased-distilled-squad', 'distilbert-base-cased', 'distilbert-base-german-cased', 'distilbert-base-multilingual-cased', 'distilbert-base-uncased-distilled-squad',
                                    'distilbert-base-uncased-finetuned-sst-2-english', 'distilbert-base-uncased', 'distilgpt2', 'distilroberta-base', 'gpt2-large', 'gpt2-medium', 'gpt2-xl', 'gpt2', 'openai-gpt', 'roberta-base-openai-detector', 'roberta-base', 'roberta-large-mnli', 'roberta-large-openai-detector', 'roberta-large', 't5-11b', 't5-3b', 't5-base', 't5-large', 't5-small', 'transfo-xl-wt103', 'xlm-clm-ende-1024', 'xlm-clm-enfr-1024', 'xlm-mlm-100-1280', 'xlm-mlm-17-1280', 'xlm-mlm-en-2048', 'xlm-mlm-ende-1024', 'xlm-mlm-enfr-1024', 'xlm-mlm-enro-1024', 'xlm-mlm-tlm-xnli15-1024', 'xlm-mlm-xnli15-1024', 'xlm-roberta-base', 'xlm-roberta-large-finetuned-conll02-dutch', 'xlm-roberta-large-finetuned-conll02-spanish', 'xlm-roberta-large-finetuned-conll03-english', 'xlm-roberta-large-finetuned-conll03-german', 'xlm-roberta-large', 'xlnet-base-cased', 'xlnet-large-cased')

class Invalid:

    paths = (None, '', ' ', "", " ", "''", '""')

class Ignore:

    files = ('flax_model.msgpack', 'rust_model.ot', 'tf_model.h5')



def get_default_device():
    return 'cpu'

def get_default_normalization():
    return True

def get_default_seq_length():
    return 128
