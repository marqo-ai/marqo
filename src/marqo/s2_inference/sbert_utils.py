from sentence_transformers import SentenceTransformer
import numpy as np
from torch import nn

from marqo.s2_inference.types import *

from marqo.s2_inference.logger import get_logger
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
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

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