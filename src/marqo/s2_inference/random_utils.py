# implements a 'model' that returns a random vector, irrespective of 
# input. does not require a model and is completely random
# used for testing purposes
from marqo.s2_inference.sbert_utils import Model
from marqo.s2_inference.types import Union, FloatTensor, List, ndarray

import numpy as np
import hashlib

def sentence_to_hash(sentence):
    return int(hashlib.sha256(sentence.encode('utf-8')).hexdigest(), 16) % 10**8

class Random(Model):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def load(self) -> None:
        self.model = None

    @staticmethod
    def _get_sentence_hash(sentence):
        return sentence_to_hash(sentence)

    def _get_sentences_hash(self, sentences):
        hashes = 0
        i= 0
        for i,sentence in enumerate(sentences):
            hashes += self._get_sentence_hash(sentence)
        return hashes // (i + 1)

    def encode(self, sentence: Union[str, List[str]], normalize: bool = True, **kwargs) -> Union[FloatTensor, ndarray]:

        if self.embedding_dimension is None or self.embedding_dimension == 0:
            raise ValueError('invalid embedding dimension size. check the model registry is correct')

        if isinstance(sentence, str):
            # same strings generate same random vector (I think)
            seed = self._get_sentence_hash(sentence)
            np.random.seed(seed)

            return np.random.rand(1, self.embedding_dimension)
        else:
            if len(sentence) == 0:
                raise ValueError('recevied empty sentence')

            seed = self._get_sentences_hash(sentence)
            np.random.seed(seed)

            return np.random.rand(len(sentence), self.embedding_dimension)
