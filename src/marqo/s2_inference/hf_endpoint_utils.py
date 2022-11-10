import requests
import numpy as np

from marqo.s2_inference.sbert_utils import Model
from marqo.s2_inference.types import Union, FloatTensor, List

from marqo.s2_inference.logger import get_logger

logger = get_logger(__name__)


class HF_ENDPOINT(Model):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def load(self) -> None:

        pass

    def encode(self, sentence: Union[str, List[str]], normalize=True, **kwargs) -> Union[FloatTensor, np.ndarray]:
        if isinstance(sentence, str):
            sentence = [sentence]

        headers = {'Authorization': f'Bearer {self._get_api_token()}'}


        json_data = {"inputs": sentence}

        res = requests.post(
            self.endpoint_url,
            headers=headers,
            json=json_data
        )

        return res.json()['embeddings']

    def _get_api_token(self) -> str:
        return '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'

