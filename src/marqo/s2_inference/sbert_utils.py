from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from marqo.s2_inference.types import *

from marqo.s2_inference.logger import get_logger

import tritonclient.http
from tritonclient.utils import triton_to_np_dtype
from tritonclient.utils import InferenceServerException
from transformers import AutoTokenizer

logger = get_logger(__name__)


class Model:
    """ generic model wrapper class
    """
    def __init__(self, model_name: str, device: str = 'cpu', batch_size: int = 2048, embedding_dim=None, max_seq_length=None) -> None:

        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dimension = embedding_dim
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

    def load(self) -> None:
        """ method to load the model
        """
        pass

    def encode(self, sentence: Union[str, List[str]]) -> None:
        """encodes a single sentece or multiple sentences

        Args:
            sentence (str): _description_
        """
        pass

class SBERT_TRITON(Model):
    """class for SBERT models

    Args:
        Model (_type_): _description_
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = None
        self.triton_client = None
        self.model_metadata = None
        self.model_config = None
        self.model_version = None
        self.triton_model_name = None

    def load(self, model_version:str= '1', url:str='localhost:8000') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/' + self.model_name)
        # print(self.model_name)
        self.triton_model_name = self.model_name + '-' + self.device
        self.model_version = model_version

        self.triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
        self.model_metadata = self.triton_client.get_model_metadata(model_name=self.triton_model_name, model_version=model_version)
        self.model_config = self.triton_client.get_model_config(model_name=self.triton_model_name, model_version=model_version)
    
    def mean_pooling(self, token_embeddings: torch.FloatTensor, attention_mask: torch.FloatTensor) -> torch.FloatTensor:
        """performs mean average pooling over a matrix with a supplied attention mask
        Args:
            token_embeddings (FloatTensor): _description_
            attention_mask (FloatTensor): _description_
        Returns:
            FloatTensor: _description_
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



    def encode(self, sentence: Union[str, List[str]], normalize = True, **kwargs) -> Union[FloatTensor, np.ndarray]:

        if self.tokenizer is None:
            self.load()

        if isinstance(sentence, str):
            sentence = [sentence]

        # seemed inconsistent with the normalization, = False, roll own

        encoded_input = self.tokenizer(sentence, padding='max_length', max_length=self.max_seq_length, truncation=True, return_tensors='pt')

        input_ids = np.array(encoded_input['input_ids'], dtype=np.int32).reshape(1, 256)
        mask = np.array(encoded_input['attention_mask'], dtype=np.int32).reshape(1, 256)

        input0 = tritonclient.http.InferInput('input__0', (1,  256), 'INT32')
        input0.set_data_from_numpy(input_ids, binary_data=False)
        input1 = tritonclient.http.InferInput('input__1', (1, 256), 'INT32')
        input1.set_data_from_numpy(mask, binary_data=False)
        
        output = tritonclient.http.InferRequestedOutput('output__0',  binary_data=False)
        response = self.triton_client.infer(self.triton_model_name, model_version=self.model_version, inputs=[input0, input1], outputs=[output])

        embeddings = response.as_numpy('output__0')

        embeddings = self.mean_pooling(FloatTensor(embeddings), FloatTensor(mask))


        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

class SBERT(Model):
    """class for SBERT models

    Args:
        Model (_type_): _description_
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def load(self) -> None:
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # if one provided, overrite
        if self.max_seq_length is None:
            self.max_seq_length = self.model.max_seq_length
        else:
            self.model.max_seq_length = self.max_seq_length


    def encode(self, sentence: Union[str, List[str]], normalize = True, **kwargs) -> Union[FloatTensor, np.ndarray]:

        if self.model is None:
            self.load()

        if isinstance(sentence, str):
            sentence = [sentence]

        # seemed inconsistent with the normalization, = False, roll own
        embeddings = self.model.encode(sentence, batch_size=self.batch_size, 
                        normalize_embeddings=False, convert_to_tensor=True)

        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

class TEST(Model):
    """class for SBERT models

    Args:
        Model (_type_): _description_
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def load(self) -> None:
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, sentence: Union[str, List[str]], **kwargs) -> Union[FloatTensor, np.ndarray]:

        if self.model is None:
            self.load()

        if isinstance(sentence, str):
            sentence = [sentence]

        return self.model.encode(sentence, batch_size=self.batch_size, **kwargs)[:, :16]