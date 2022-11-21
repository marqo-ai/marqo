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
from marqo.s2_inference.s2_inference import _create_model_cache_key, _float_tensor_to_list, _nd_array_to_list
from marqo.s2_inference.configs import ModelCache

from marqo.s2_inference.logger import get_logger
logger = get_logger(__name__)

def _convert_cross_encoder_output(output: Union[FloatTensor, ndarray, List[float]]) -> List[float]:
    """converts the model outputs to a list of floats

    Args:
        output (FloatTensor): _description_

    Returns:
        List[List[float]]: _description_
    """

    if _verify_model_outputs(output):
        return output

    if isinstance(output, (FloatTensor, Tensor)):
        output = output.squeeze()
        output = _float_tensor_to_list(output)
    
    elif isinstance(output, ndarray):
        output = output.squeeze()
        output = _nd_array_to_list(output)

    elif isinstance(output, list):
        if isinstance(output[0], FloatTensor):
            output = [_float_tensor_to_list(_o) for _o in output]
        elif isinstance(output[0], ndarray):
            output = [_nd_array_to_list(_o) for _o in output]
        else:
            raise TypeError(f"unsupported nested list with elements of type {type(output[0])}")

    else:
        raise TypeError(f"unsupported output type of {type(output)}")

    if _verify_model_outputs(output):
        return output
    
    raise TypeError(f"unable to convert input of type {type(output)} to a list of lists of floats")


def _verify_model_outputs(outputs: Union[List, ndarray,FloatTensor]) -> bool:
    """checks the outpus conforms to the standard of a list of floats or ints

    Args:
        outputs (Union[List, ndarray,FloatTensor]): _description_

    Raises:
        TypeError: _description_

    Returns:
        bool: _description_
    """
    if not isinstance(outputs, list):
        return False

    if len(outputs) == 0:
        return True

    if isinstance(outputs[0], (ndarray, FloatTensor ,list)):
        return False

    if isinstance(outputs[0], (float, int)):
        return True

    raise TypeError(f"unknown output type of {type(outputs)} and {type(outputs[0])} expected list")
    
def _verify_model_inputs(list_of_lists: List[List]) -> bool:
    """check the format of the model inputs

    Args:
        list_of_lists (List[List]): _description_

    Returns:
        bool: _description_
    """
    return all(isinstance(x, (list, tuple)) for x in list_of_lists)

def convert_device_id_to_int(device: str = 'cpu'):
    """maps the string device, 'cpu', 'cuda', 'cuda:#'
    to an int for HF pipelines device representation

    Args:
        device (str, optional): _description_. Defaults to 'cpu'.

    Raises:
        ValueError: _description_
        TypeError: _description_

    Returns:
        _type_: _description_
    """
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
    """ used for mocking the model

    Returns:
        _type_: _description_
    """
    def __init__(self, *args, **kwargs) -> None:
        pass

    def predict(self, inputs: Iterable):

        return np.random.rand(len(inputs))

class HFClassificationOnnx:
    """uses HF pipelines and optimum to load hf classification model 
    (cross encoders) and uses it as onnx
    https://huggingface.co/docs/optimum/main/en/onnxruntime/modeling_ort
    
    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    
    def __init__(self, model_name: str, device: str = 'cpu', max_length: int = 512) -> None:

        self.model_name = model_name
        self.save_path = None
        self.device_string = device
        self.device = convert_device_id_to_int(device)
        self.max_length = max_length
        self.tokenizer_kwargs = {'padding':True, 'truncation':True,  'max_length':self.max_length}

        # TODO load local version
        #self.load_from_cache = load_from_cache

        self.model = ORTModelForSequenceClassification.from_pretrained(self.model_name, from_transformers=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.onnx_classifier = pipeline("text-classification", model=self.model, 
                                        tokenizer=self.tokenizer, device=self.device)

    def _get_save_name(self) -> None:
        """generates the saeve name for local storage
        """
        self.save_path = os.path.join(ModelCache.onnx_cache_path, self.model_name + '_onnx')
        self.model_save_name = self.save_path
        self.tokenizer_save_name = self.save_path

    def save(self) -> None:
        """saves the model locally
        """
        logger.info(f"saving model to {self.model_save_name}")
        if self.save_path is None:
            self._get_save_name()

        self.model.save_pretrained(self.model_save_name)
        self.tokenizer.save_pretrained(self.tokenizer_save_name)

    @staticmethod
    def _prepare_inputs(inputs: List[List[str]]) -> List[Dict]:
        """prepares the inputs for the onnx cross encoder
        used named fields in the output to allow for proper passing of two strings

        Args:
            inputs (List[List[str]]): _description_

        Raises:
            RuntimeError: _description_

        Returns:
            List[Dict]: _description_
        """
        if not _verify_model_inputs(inputs):
            raise RuntimeError(f"expected list of lists, received {type(inputs)} of {type(inputs[0])}")

        return [{'text':pair[0], 'text_pair':pair[1]} for pair in inputs]

    @staticmethod
    def _parepare_outputs(outputs: Dict) -> ndarray:
        """takes the outputs of the onnx model (dict) and extracts the score
        assumes binary labels
        Args:
            outputs (Dict): _description_

        Returns:
            ndarray: _description_
        """
        return [float(pred['score']) for pred in outputs]

    def predict(self, inputs: List[Dict]) -> List[Dict]:
        """onnx predict method

        Args:
            inputs (List[Dict]): _description_

        Returns:
            List[Dict]: _description_
        """
        self.inputs = self._prepare_inputs(inputs)
        # couldn't find aaaaany documentation on passing tokenizer arguments through the pipeline
        # leaving these here for reference
        # https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/__init__.py#L750
        # https://stackoverflow.com/questions/67849833/how-to-truncate-input-in-the-huggingface-pipeline
        self.predictions = self.onnx_classifier(self.inputs, **self.tokenizer_kwargs)
        self.outputs = self._parepare_outputs(self.predictions)

        return self.outputs


def load_sbert_cross_encoder_model(model_name: str, device: str = 'cpu', max_length: int = 512) -> Dict:
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
            model = CrossEncoder(model_name, max_length=max_length, device=device, default_activation_function=torch.nn.Sigmoid())
            if hasattr(model.tokenizer, 'model_max_length'):
                model_max_len = model.tokenizer.model_max_length
                if max_length > model_max_len:
                    model.max_length = model_max_len
                    logger.warning(f"specified max_length of {max_length} is greater than model max length of {model_max_len}, setting to model max length")
        available_models[model_cache_key] = model

    return {'model':model}


def load_hf_cross_encoder_model(model_name: str, device: str = 'cpu') -> Dict:
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

def load_owl_vit(model_name: str, device: str = 'cpu') -> Dict:
    """loader for owl vit for image reranking

    Args:
        model_name (str): _description_
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        Dict: _description_
    """

    model_cache_key = (model_name, device)

    if model_cache_key in available_models:
        logger.info(f"loading {model_cache_key} from cache...")
        model, processor = available_models[model_cache_key] 
    else:
        processor = OwlViTProcessor.from_pretrained(model_name)
        model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)
        available_models[model_cache_key] = model, processor

    model.eval()

    # TODO use a small class to store the different model pieces and configs
    return {'model':model, 'processor':processor}

def _process_owl_inputs(processor, texts, images):
    """wrapper for processing the owl inputs

    Args:
        processor (_type_): _description_
        texts (_type_): _description_
        images (_type_): _description_

    Returns:
        _type_: _description_
    """
    return processor(text=texts, images=images, return_tensors="pt")

def _predict_owl(model, processed_inputs, post_process_function, size):
    """helper to predict with owl

    Args:
        model (_type_): _description_
        processed_inputs (_type_): _description_
        post_process_function (_type_): _description_
        size (_type_): _description_

    Returns:
        _type_: _description_
    """
    with torch.no_grad():
        outputs = model(**processed_inputs)
        # there is a bug in the hf code https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/owlvit/feature_extraction_owlvit.py#L140
        outputs.logits = outputs.logits.to('cpu')
        outputs.pred_boxes = outputs.pred_boxes.to('cpu')
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([size[::-1]])
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = post_process_function(outputs=outputs, target_sizes=target_sizes)
        return results

def process_owl_results(results: List) -> List:
    """wrapper for processing a list of resultsd from owl-vitexi

    Args:
        results (_type_): _description_

    Returns:
        _type_: _description_
    """
    rezs = []
    for result in results:
        rez = _process_owl_result(result)
        rezs.append(rez)
    return rezs

def _process_owl_result(result: List[Dict], identifier: str) -> Tuple[List, List, List]:
    """post-process the owl-vit results

    Args:
        result (List[Dict]): _description_
        identifier (str): _description_

    Returns:
        Tuple[List, List, List]: _description_
    """
    boxes, scores, _ = result[0]["boxes"], result[0]["scores"], result[0]["labels"]
 
    boxes_round = []
    for i in range(len(boxes)):
        boxes_round.append([round(i, 2) for i in boxes[i].tolist()])

    return boxes, scores, [identifier]*len(scores)

def sort_owl_boxes_scores(boxes: List, scores: List, identifier: List) -> Tuple[List, List, List]:
    """sorts the lists based on  the scores

    Args:
        boxes (List): _description_
        scores (List): _description_
        identifier (List): _description_

    Raises:
        RuntimeError: _description_
        RuntimeError: _description_

    Returns:
        Tuple[List, List, List]: _description_
    """
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

    return boxes, scores, identifier

def _keep_top_k(input_list: List, k: int = 1):
    """return top-k of a list

    Args:
        input_list (List): _description_
        k (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    return input_list[:k]
