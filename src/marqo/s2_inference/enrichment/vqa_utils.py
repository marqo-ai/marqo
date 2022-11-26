from lavis.models import load_model_and_preprocess

from marqo.s2_inference.types import *
from marqo.s2_inference.errors import EnrichmentError
from marqo.s2_inference.logger import get_logger
from marqo.s2_inference.clip_utils import format_and_load_CLIP_images
from marqo.s2_inference.enrichment import enums

logger = get_logger("enrichment")

def get_allowed_vqa_tasks():
    
    return {
            "attribute-extraction":{"name":'blip_vqa', 'model_type':'vqav2'}, 
            "question-answer":{"name":'blip_vqa', 'model_type':'vqav2'}}

class VQA:
    
    # these basically map out the tasks
    mappings = get_allowed_vqa_tasks()

    def __init__(self, device: str = 'cpu', **kwargs) -> None:
        
        self.device = device

        self.name_and_model = {"name":'blip_vqa', 'model_type':'vqav2'}

        self.model = None
        self.vis_processors = None
        self.txt_processors = None
        self.answers = None
        self.kwargs = kwargs

    def load(self):
        logger.info(f"loading {self.name_and_model} using {self.name_and_model['name']} and {self.name_and_model['model_type']}")
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name=self.name_and_model['name'], 
                                                        model_type=self.name_and_model['model_type'], 
                                                        is_eval=True, device=self.device)

    def predict(self, task, *args, **kwargs):
        """
        getting called once per doc that might itself have multiple questions
      
        {
                "attributes": ["Bathroom, Bedroom, Study, Yard"]
                "image_field": "https://s3.image.png"
            },


        Args:
            task (_type_): _description_
        """

        self.task = task
        if self.task not in self.mappings:
            raise EnrichmentError(f"incorrect enrichment type specified {self.task} expected one of {list(self.mappings.keys())}")

        image_query_pairs = _process_kwargs_vqa(kwargs)
        self._infer(self.task, image_query_pairs)

        return self.answers

    @staticmethod
    def _load_images(image_names: List[str]):
        images = format_and_load_CLIP_images(image_names)
        return images

    @staticmethod
    def _format_questions_for_task(task: str, questions: List[str]) -> List[str]:
        if task == "attribute-extraction":
            template = get_attribute_extraction_template()
            return [template.format(q) for q in questions] 
        else:
            return list(questions)


    def _infer(self, task, image_query_pairs):

        image_names, questions = zip(*image_query_pairs)

        # optionally modify question based on task
        questions = self._format_questions_for_task(task, questions)
        # preload the images
        images = self._load_images(image_names=list(image_names))

        # loop through and infer - can batch later
        answers = []
        for qi,(raw_image,question) in enumerate(zip(images,questions)):
            
            image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
            question = self.txt_processors["eval"](question)
            result = self.model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")[0]
            answers.append(result)

        # optionally convert answers dpending on task
        self.answers = _format_answers_for_task(task, answers)

        # put in a check for now on output type
        if not _check_output_type(self.answers):
            raise EnrichmentError(f"incorrect output type found {type(self.answers)} for {self.answers}")

def _format_answers_for_task(task, answers):
    
    if task == "attribute-extraction":
        return [convert_to_bool(a) for a in answers] 
    else:
        return answers


def _check_output_type(output: List[Union[str, bool, int, float]]) -> bool:
    type_correct = False
    if isinstance(output, List):
        if len(output) == []:
            return True
        type_correct = all(isinstance(v, (str, float, int)) for v in output)
    return type_correct

def get_attribute_extraction_template() -> str:
    return "does this picture have a {} in it?"

def convert_to_bool(str_in: str, positives: List[Any] = ['yes', True, 'True', 'true', 1, '1', 'y', 't']) -> bool:
    if str(str_in).lower() in positives:
        return True
    return False 

def _process_kwargs_vqa(kwargs_dict: Dict) -> Tuple[List[Any], List[Any]]:
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
    # return list_of_images, list_of_queries


