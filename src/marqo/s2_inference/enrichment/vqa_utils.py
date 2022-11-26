from lavis.models import load_model_and_preprocess

from marqo.s2_inference.types import *
from marqo.s2_inference.errors import EnrichmentError
from marqo.s2_inference.logger import get_logger

from marqo.s2_inference.enrichment import enums

logger = get_logger("enrichment")

def get_allowed_vqa_tasks():
    
    return {
            "attribute-extraction":{"name":'blip_vqa', 'model_type':'vqav2'}, 
            "question-answer":{"name":'blip_vqa', 'model_type':'vqav2'}}

class VQA:
    
    # these basically map out the tasks
    mappings = get_allowed_vqa_tasks()

    def __init__(self, model: str = "vqa", device: str = 'cpu', **kwargs) -> None:

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
         """
     getting called once per doc that might itself have multiple questions
     {
            "attributes": [{"string": "Bathroom, Bedroom, Study, Yard"} ]
            "image_field": {'document_field":"Image location"}
     	},
     	->
    {
            "attributes": ["Bathroom, Bedroom, Study, Yard"]
            "image_field": "https://s3.image.png"
     	},


        Args:
            task (_type_): _description_
        """

        

    # def _infer(self, )

def _process_kwargs_vqa(kwargs_dict: Dict) -> Dict:
    """_summary_
    {
                "attributes": ["Bathroom, Bedroom, Study, Yard"]
                "image_field": "https://s3.image.png"
            },
    -> 
    [ 
        ("https://s3.image.png", "Bathroom"),
        ("https://s3.image.png", "Bedroom"),
        ("https://s3.image.png", "Study"),
        ("https://s3.image.png", "Yard"),
    ]
    Args:
        kwargs_dict (Dict): _description_

    Raises:
        EnrichmentError: _description_

    Returns:
        Dict: _description_
    """

    if enums.VQA_kwargs.image_field not in kwargs_dict:
        raise EnrichmentError(f"found {list(kwargs_dict.keys())} but expected {enums.VQA_kwargs.image_field}")

    image_field = kwargs_dict[enums.VQA_kwargs.image_field]

    if not isinstance(image_field, str):
        raise EnrichmentError(f"wrong type for image_field found {image_field} expected string")

    if enums.VQA_kwargs.attributes in kwargs_dict:
        attributes = kwargs_dict[enums.VQA_kwargs.attributes]

        # TODO seperate validation function
        if isinstance(attributes, (list, str)):
            if isinstance(attributes, list):
                if len(attributes) != 1:
                    raise EnrichmentError(f"found list of wrong size for attributes {attributes}")
                attributes = attributes[0]
        else:
            raise EnrichmentError(f"wrong type for attributes {attributes}")

        list_of_queries = attributes.split(enums.VQA_kwargs.attribute_seperator)
        list_of_queries = [l.strip() for l in list_of_queries]

    elif enums.VQA_kwargs.query in kwargs_dict:
        query = kwargs_dict[enums.VQA_kwargs.query]
        # should be list of str or str
        if isinstance(query, (str, list)):
            if isinstance(query, str):
                list_of_queries = [query]
            else:
                list_of_queries = query
        else:
            raise EnrichmentError(f"wrong type for attributes {query}")
    else:
        raise EnrichmentError(f"found {list(kwargs_dict.keys())} but expected {enums.VQA_kwargs.attributes} or {enums.VQA_kwargs.query}")

    # now we should have a list of queries and a single image
    # we create the pairs for inferencing
    N = len(list_of_queries)
    list_of_images = [image_field]*N

    image_query_pairs = list(zip(list_of_images, list_of_queries))

    return image_query_pairs


# class CLIP:
    
#     """
#     conveniance class wrapper to make clip work easily for both text and image encoding
#     """

#     def __init__(self, model_type: str = "ViT-B/32", device: str = 'cpu',  embedding_dim: int = None,
#                             truncate: bool = True, **kwargs) -> None:

#         self.model_type = model_type
#         self.device = device
#         self.model = None
#         self.tokenizer = None
#         self.processor = None
#         self.embedding_dimension = embedding_dim
#         self.truncate = truncate

#     def load(self) -> None:

#         # https://github.com/openai/CLIP/issues/30
#         self.model, self.preprocess = clip.load(self.model_type, device='cpu', jit=False)
#         self.model = self.model.to(self.device)
#         self.tokenizer = clip.tokenize
#         self.model.eval()
    
#     def _convert_output(self, output):

#         if self.device == 'cpu':
#             return output.numpy()
#         elif self.device.startswith('cuda'):
#             return output.cpu().numpy()

#     @staticmethod
#     def normalize(outputs):
#         return outputs.norm(dim=-1, keepdim=True)

#     def encode_text(self, sentence: Union[str, List[str]], normalize = True) -> FloatTensor:
        
#         if self.model is None:
#             self.load()
        
#         text = self.tokenizer(sentence, truncate=self.truncate).to(self.device)

#         with torch.no_grad():
#             outputs =  self.model.encode_text(text)

#         if normalize:
#             _shape_before = outputs.shape
#             outputs /= self.normalize(outputs)
#             assert outputs.shape == _shape_before

#         return self._convert_output(outputs)

#     def encode_image(self, images: Union[str, ImageType, List[Union[str, ImageType]]], 
#                         normalize = True) -> FloatTensor:
        
#         if self.model is None:
#             self.load()

#         # default to batch encoding
#         if isinstance(images, list):
#             image_input = format_and_load_CLIP_images(images)
#         else:
#             image_input = [format_and_load_CLIP_image(images)]

#         self.image_input_processed = torch.stack([self.preprocess(_img).to(self.device) for _img in image_input])
    
#         with torch.no_grad():
#             outputs = self.model.encode_image(self.image_input_processed)

#         if normalize:
#             _shape_before = outputs.shape
#             outputs /= self.normalize(outputs)
#             assert outputs.shape == _shape_before
#         return self._convert_output(outputs)

#     def encode(self, inputs: Union[str, ImageType, List[Union[str, ImageType]]], 
#                                 default: str = 'text', normalize = True, **kwargs) -> FloatTensor:

#         infer = kwargs.pop('infer', True)

#         if infer and _is_image(inputs):
#             is_image = True
#         else:
#             is_image = False
#             if default == 'text':
#                 is_image = False
#             elif default == 'image':
#                 is_image = True
#             else:
#                 raise ValueError(f"expected default='image' or default='text' but received {default}")

#         if is_image:
#             logger.debug('image')
#             return self.encode_image(inputs, normalize=normalize)
#         else:
#             logger.debug('text')
#             return self.encode_text(inputs, normalize=normalize)