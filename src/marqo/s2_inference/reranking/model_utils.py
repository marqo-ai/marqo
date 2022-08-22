from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from marqo.s2_inference.types import *
from marqo.s2_inference.s2_inference import available_models
from marqo.s2_inference.s2_inference import _create_model_cache_key

from marqo.s2_inference.logger import get_logger
logger = get_logger(__name__)


def load_sbert_cross_encoder_model(model_name: str, device: str = 'cpu', max_length: int = 512) -> Any:
    """    
    
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
        model = CrossEncoder(model_name, max_length=max_length)
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