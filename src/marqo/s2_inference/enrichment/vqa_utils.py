from lavis.models import load_model_and_preprocess

from marqo.s2_inference.errors import EnrichmentError
from marqo.s2_inference.logger import get_logger

logger = get_logger("enrichment")

def get_allowed_vqa_tasks():
    
    return {
            "attribute-extraction":{"name":'blip_vqa', 'model_type':'vqav2'}, 
            "question-answer":{"name":'blip_vqa', 'model_type':'vqav2'}}

class VQA:
    
    # these basically map out the tasks
    mappings = get_allowed_vqa_tasks()

    def __init__(self, task: str = "attribute-extraction", device: str = 'cpu', **kwargs) -> None:

        self.task = task
        if self.task not in self.mappings:
            raise EnrichmentError(f"incorrect enrichment type specified {self.task} expected one of {list(self.mappings.keys())}")
        self.device = device

        self.name_and_model = self.mappings[self.task]

        self.model = None
        self.vis_processors = None
        self.txt_processors = None

    def load(self):
        logger.info(f"loading {self.model_type} using {self.name_and_model['name']} and {self.name_and_model['model']}")
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name=self.name_and_model['name'], 
                                                        model_type=self.name_and_model['model_type'], 
                                                        is_eval=True, device=self.device)

    def predict(self, task, *args, **kwargs):


class CLIP:
    
    """
    conveniance class wrapper to make clip work easily for both text and image encoding
    """

    def __init__(self, model_type: str = "ViT-B/32", device: str = 'cpu',  embedding_dim: int = None,
                            truncate: bool = True, **kwargs) -> None:

        self.model_type = model_type
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.embedding_dimension = embedding_dim
        self.truncate = truncate

    def load(self) -> None:

        # https://github.com/openai/CLIP/issues/30
        self.model, self.preprocess = clip.load(self.model_type, device='cpu', jit=False)
        self.model = self.model.to(self.device)
        self.tokenizer = clip.tokenize
        self.model.eval()
    
    def _convert_output(self, output):

        if self.device == 'cpu':
            return output.numpy()
        elif self.device.startswith('cuda'):
            return output.cpu().numpy()

    @staticmethod
    def normalize(outputs):
        return outputs.norm(dim=-1, keepdim=True)

    def encode_text(self, sentence: Union[str, List[str]], normalize = True) -> FloatTensor:
        
        if self.model is None:
            self.load()
        
        text = self.tokenizer(sentence, truncate=self.truncate).to(self.device)

        with torch.no_grad():
            outputs =  self.model.encode_text(text)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before

        return self._convert_output(outputs)

    def encode_image(self, images: Union[str, ImageType, List[Union[str, ImageType]]], 
                        normalize = True) -> FloatTensor:
        
        if self.model is None:
            self.load()

        # default to batch encoding
        if isinstance(images, list):
            image_input = format_and_load_CLIP_images(images)
        else:
            image_input = [format_and_load_CLIP_image(images)]

        self.image_input_processed = torch.stack([self.preprocess(_img).to(self.device) for _img in image_input])
    
        with torch.no_grad():
            outputs = self.model.encode_image(self.image_input_processed)

        if normalize:
            _shape_before = outputs.shape
            outputs /= self.normalize(outputs)
            assert outputs.shape == _shape_before
        return self._convert_output(outputs)

    def encode(self, inputs: Union[str, ImageType, List[Union[str, ImageType]]], 
                                default: str = 'text', normalize = True, **kwargs) -> FloatTensor:

        infer = kwargs.pop('infer', True)

        if infer and _is_image(inputs):
            is_image = True
        else:
            is_image = False
            if default == 'text':
                is_image = False
            elif default == 'image':
                is_image = True
            else:
                raise ValueError(f"expected default='image' or default='text' but received {default}")

        if is_image:
            logger.debug('image')
            return self.encode_image(inputs, normalize=normalize)
        else:
            logger.debug('text')
            return self.encode_text(inputs, normalize=normalize)