import numpy as np
import os

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    OwlViTProcessor, 
    OwlViTForObjectDetection,
    pipeline,
)

from optimum.onnxruntime import ORTModelForSequenceClassification
from sentence_transformers import CrossEncoder
import torch

from marqo.s2_inference.types import *
from marqo.s2_inference.s2_inference import available_models
from marqo.s2_inference.s2_inference import _create_model_cache_key
from marqo.s2_inference.configs import ModelCache

from marqo.s2_inference.logger import get_logger
logger = get_logger(__name__)


def _verify_model_inputs(list_of_lists):
    return all(isinstance(x, (list, tuple)) for x in list_of_lists)

def convert_device_id_to_int(device: str = 'cpu'):

    if device[:4] not in ['cpu', 'cuda']:
        raise ValueError(f"expected one of cpu or cuda or cuda:# but received {device}")

    if device == 'cpu':
        return -1
    
    if device == 'cuda':
        return 0

    if device.startswith('cuda:'):
        # check if it is id'd by number
        if device[-1].isnumeric():
            return int(device.replace('cuda:', ''))
    
    raise TypeError(f"unexpected device {device}")

class DummyModel:
    # used for mocking the model
    def __init__(self, *args, **kwargs) -> None:
        pass

    def predict(self, inputs):

        return np.random.rand(len(inputs))

class HFClassificationOnnx:
    # https://huggingface.co/docs/optimum/main/en/onnxruntime/modeling_ort

    def __init__(self, model_name: str, device: str = 'cpu', load_from_cache: bool = True):

        self.model_name = model_name
        self.save_path = None
        self.device_string = device
        self.device = convert_device_id_to_int(device)

        # TODO load local version
        self.model = ORTModelForSequenceClassification.from_pretrained(self.model_name, from_transformers=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.onnx_classifier = pipeline("text-classification", model=self.model, 
                                        tokenizer=self.tokenizer, device=self.device)


    def _get_save_name(self):

        self.save_path = os.path.join(ModelCache.onnx_cache_path, self.model_name + '_onnx')
        self.model_save_name = self.save_path
        self.tokenizer_save_name = self.save_path

    def save(self):

        logger.info(f"saving model to {self.model_save_name}")
        if self.save_path is None:
            self._get_save_name()

        self.model.save_pretrained(self.model_save_name)
        self.tokenizer.save_pretrained(self.tokenizer_save_name)

    @staticmethod
    def _prepare_inputs(inputs: List[List[str]]):
        if not _verify_model_inputs(inputs):
            raise RuntimeError(f"expected list of lists, received {type(inputs)} of {type(inputs[0])}")

        return [{'text':pair[0], 'text_pair':pair[1]} for pair in inputs]

    @staticmethod
    def _parepare_outputs(outputs):
        return np.array([pred['score'] for pred in outputs])

    def predict(self, inputs):
        self.inputs = self._prepare_inputs(inputs)
        self.predictions = self.onnx_classifier(self.inputs)
        self.outputs = self._parepare_outputs(self.predictions)

        return self.outputs


def load_sbert_cross_encoder_model(model_name: str, device: str = 'cpu', max_length: int = 512) -> Any:
    """    
    https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-2
    scores = model.predict([('Query', 'Paragraph1'), ('Query', 'Paragraph2') , ('Query', 'Paragraph3')])

    Args:
        model_name (str): _description_

    Returns:
        Any: _description_
    """
    model_cache_key = _create_model_cache_key(model_name, device)

    if model_cache_key in available_models:
        model = available_models[model_cache_key] 
    else:
        logger.info(f"loading {model_name} on device {device} and adding to cache...")
        if model_name == '_testing':
            model = DummyModel()
            logger.warning('using the test model - << TESTING PURPOSES ONLY >>')
        elif model_name.startswith('onnx/'):
            model = HFClassificationOnnx(model_name.replace('onnx/', ''), device=device)
        else:
            model = CrossEncoder(model_name, max_length=max_length, device=device)
        available_models[model_cache_key] = model

    return {'model':model}


def load_hf_cross_encoder_model(model_name: str, device: str = 'cpu') -> Tuple[Any, Any]:
    """    
    
    features = tokenizer(['How many people live in Berlin?', 'How many people live in Berlin?'], ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.'],  padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(**features).logits

    Args:
        model_name (str): _description_

    Returns:
        Any: _description_
    """

    model_cache_key = _create_model_cache_key(model_name, device)

    if model_cache_key in available_models:
        model, tokenizer = available_models[model_cache_key] 
    else:
        logger.info(f"loading {model_name} on device {device} and adding to cache...")    
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.eval()
    
    return {'model':model, 'tokenizer':tokenizer}

def load_owl_vit(device: str = 'cpu'):
    
    if ('owl', device) in available_models:
        model, processor = available_models[('owl', device)] 
    else:
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32") #pathc16, patch
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        available_models[('owl', device)] = model, processor

    model.eval()

    # TODO use a small class to store the different model pieces and configs
    return {'model':model, 'processor':processor}

def _process_owl_inputs(processor, texts, images):
    return processor(text=texts, images=images, return_tensors="pt")

def _predict_owl(model, processed_inputs, post_process_function, size):
    
    with torch.no_grad():
        outputs = model(**processed_inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([size[::-1]])
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = post_process_function(outputs=outputs, target_sizes=target_sizes)

        return results

def process_owl_results(results):
    rezs = []
    for result in results:
        rez = _process_owl_result(result)
        rezs.append(rez)
    return rezs

def _process_owl_result(result, identifier):
    # process indiviudal result
    boxes, scores, _ = result[0]["boxes"], result[0]["scores"], result[0]["labels"]
 
    boxes_round = []
    for i in range(len(boxes)):
        boxes_round.append([round(i, 2) for i in boxes[i].tolist()])

    return boxes, scores, [identifier]*len(scores)

def sort_owl_boxes_scores(boxes, scores, identifier):

    if len(scores) != len(boxes):
        # TODO use Marqo errors 
        raise RuntimeError(f"expected each bbox to have a score. found {len(boxes)} boxes and {len(scores)} scores")

    inds = scores.argsort(descending=True)
    boxes = boxes[inds]
    scores = scores[inds]

    if identifier is not None and len(identifier) != 0:
        if len(identifier) != len(boxes):
            # TODO use Marqo errors 
            raise RuntimeError(f"expected each bbox to have an identifier. " \
                f"found {len(boxes)} boxes and {len(identifier)} identifiers")
        identifier = [identifier[i] for i in inds]

    # if images is not None and len(images) != 0:
    #     if len(images) != len(boxes):
    #         # TODO use Marqo errors 
    #         raise RuntimeError(f"expected each bbox to have an image. " \
    #             f"found {len(boxes)} boxes and {len(images)} identifiers")
    
    #     images = [images[i] for i in inds]
    return boxes, scores, identifier