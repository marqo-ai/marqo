from functools import partial
import functools
import validators
import uuid
from collections import defaultdict

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import torch

from marqo.s2_inference.types import *
from marqo.s2_inference.reranking.model_utils import (
    load_sbert_cross_encoder_model,
    load_owl_vit,
    _process_owl_inputs,
    _predict_owl,
    sort_owl_boxes_scores,
    _verify_model_inputs,
    _convert_cross_encoder_output,
    _process_owl_result,
    _keep_top_k
    )
from marqo.s2_inference.errors import RerankerNameError
from marqo.s2_inference.clip_utils import load_image_from_path
from marqo.s2_inference.reranking.enums import Columns, ResultsFields
from marqo.s2_inference.reranking.configs import get_default_text_processing_parameters
from marqo.s2_inference.processing import text as text_processor
from marqo.s2_inference.processing import image as image_processor

from marqo.s2_inference.logger import get_logger
logger = get_logger(__name__)


class FormattedResults:

    """
    helper class to format results.
    use this as the interface between results and models and reranking.
    output should be a dataframe with all fields required.
    """    
    
    def __init__(self, results: Dict, highlights_field: str = ResultsFields.highlights, searchable_fields: List[str] = None):

        self.results = results
        self.highlights_field = highlights_field
        self.searchable_fields = searchable_fields

        # check ids exist and if not create some
        self._fill_doc_ids(self.results)

        self.results_to_df()

        self._get_searchable_columns()

    def results_to_df(self) -> None:
        """ converts the results dict from search into a dataframe for easier manipulation

        Returns:
            _type_: _description_
        """

        if self.searchable_fields is not None and isinstance(self.searchable_fields, list):
            self.results[ResultsFields.hits] = [r for r in self.results[ResultsFields.hits] if all(s in r for s in self.searchable_fields)]

        self.results_df = pd.DataFrame(self.results[ResultsFields.hits])

        def _get_highlights(content):
            if content == []:
                return None
            elif isinstance(content, (dict, defaultdict)):
                _key = list(content.keys())
                if len(_key) == 0:
                    return None
                if len(_key) > 1:
                    logger.warning(f"found more than 1 highlight. found {_key}. keeping the first only...")
                return content[_key[0]]

        if self.highlights_field in self.results_df:
            self.results_df[self.highlights_field] = self.results_df[self.highlights_field].apply(_get_highlights)

    @staticmethod
    def _fill_doc_ids(results: Dict) -> None:
        """
        check if an id exists, otherwise create a temporary one for easier identification
        during re-ranking
        
        Args:
            results (Dict): _description_
        """

        for result in results[ResultsFields.hits]:
            if ResultsFields.id not in result:
                doc_id = str(uuid.uuid4())
                result[ResultsFields.reranked_id] = doc_id 
            else:
                result[ResultsFields.reranked_id] = result[ResultsFields.id]

    def _get_searchable_columns(self) -> None:
        """get the fields of the documents we can use for searching
        """
        self.searchable_fields = [field for field in self.results_df.columns.tolist() if not field.startswith('_')]

    @staticmethod
    def format_for_model(results_df: pd.DataFrame, searchable_fields: List[str], query: str = None):
        """formats the converted results dataframe into something that goes to the models. 
           self.results_to_df needs to be called to provide the input

        Args:
            results_df (pd.DataFrame): _description_
            searchable_fields (List[str]): _description_
            query (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # we want the output here to be tuples of the attribute content,query, id and attribute that was used
        # the first two will go to the model while the latter will be used to report the results
        
        # the number 1.0 is arbitrary to some degree (depends on how scores get combined)
        if ResultsFields.original_score not in results_df:
            results_df[ResultsFields.original_score] = 1.0

        inputs = []
        for field in searchable_fields:
            _inputs_df = results_df[[field, ResultsFields.reranked_id, ResultsFields.original_score]]
            _inputs_df[Columns.field_name] = field

            # this is the case if we have some documents that do not have all the fields, 
            # the empty fields go to nan when converted to pandas
            _inputs_df = _inputs_df[_inputs_df[field].notna()] 

            if len(_inputs_df) > 0:
                
                _inputs_df[Columns.query] = query

                # we keep it in case we want to do hybrid search
                # the number 1.0 is arbitrary to some degree (depends on how scores get combined)
                if ResultsFields.original_score not in _inputs_df:
                    _inputs_df[ResultsFields.original_score] = 1.0

                fields_to_keep = [Columns.query, field, ResultsFields.reranked_id, Columns.field_name, ResultsFields.original_score]

                inputs += _inputs_df[fields_to_keep].values.tolist()
            
        inputs_df = pd.DataFrame(inputs, columns=[Columns.query, Columns.field_content, ResultsFields.reranked_id, Columns.original_field_name,  ResultsFields.original_score])
        
        return inputs_df

class ReRanker:
    """base class for the rerankers
    """

    def __init__(self, ):
        pass 

    def load_model(self):
        pass

    def format_results(self, results: Dict, query: str = None, searchable_fields: List[str] = None):
        """standardize the way the results are formatted to go to a standard cross-encoder

        Args:
            results (Dict): _description_
            query (str, optional): _description_. Defaults to None.
        """
        self.results = results
        self.formatted_results = FormattedResults(self.results, searchable_fields=searchable_fields)

    @staticmethod
    def _prepare_inputs(inputs_df: pd.DataFrame, query_column: str = Columns.query, 
                            content_column: str = Columns.field_content) -> List[List[str]]:
        """subselects the columns from the formatted dataframe and converts to a list
        for feeding to the cross encoder

        Args:
            inputs_df (pd.DataFrame): _description_
            query_column (str, optional): _description_. Defaults to Columns.query.
            content_column (str, optional): _description_. Defaults to Columns.field_content.

        Returns:
            pd.DataFrame: _description_
        """
        return inputs_df[[query_column, content_column]].values.tolist()

    def get_reranked_results(self, score_column: str = ResultsFields.reranker_score, 
                            highlight_content_column: str = Columns.field_content):
        """reranks the updated results using the score in score_column

        Args:
            score_column (str, optional): _description_. Defaults to ResultsFields.reranker_score.
        """
        reranked_top = []
        for i,group in self.inputs_df.groupby(ResultsFields.reranked_id):
            group = group.sort_values(score_column, ascending=False).head(self.num_highlights)
            reranked_top.append(group)
        reranked_top = pd.concat(reranked_top).set_index(ResultsFields.reranked_id)
        
        # find out which document is the highest, then do the field as a highlight
        for result in self.results[ResultsFields.hits]:
            rerank_id = result[ResultsFields.reranked_id]
            if self.num_highlights == 1:
                _score = reranked_top.loc[rerank_id][score_column]
            else:
                _score = reranked_top.loc[rerank_id][score_column].values.tolist()
                
            result[ResultsFields.reranker_score] = _score
            _df = reranked_top.loc[rerank_id]
             
            if self.num_highlights == 1:
                _content = [{_df[Columns.original_field_name]:_df[highlight_content_column]}]
            else:
                _content = [{row[Columns.original_field_name]:row[highlight_content_column]} for _, row in _df.iterrows()]

            result[ResultsFields.highlights_reranked] = _content

        self.results[ResultsFields.hits] = sorted(self.results[ResultsFields.hits], key=lambda x:x[ResultsFields.reranker_score], reverse=True)

    def rerank(self, query, results):
        # this gets filled on for the task (text/images)
        pass


class ReRankerText(ReRanker):
    """
    class for reranking with hf based text models
    """
    def __init__(self, model_name: str, device: str, max_length: int = 512, num_highlights: int = 1, 
                        split_params=get_default_text_processing_parameters()):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.num_highlights = num_highlights
        self.formatted_results = None
        self.split_params = split_params
        self.model = None

        self.split_length = None
        self.split_overlap = None
        self.split_method = None

        if self.split_params is not None and isinstance(self.split_params, (dict, defaultdict)):
            self._extract_text_processing_parameters()

    def _extract_text_processing_parameters(self) -> None:
        self.split_length = self.split_params.splitLength
        self.split_overlap = self.split_params.splitOverlap
        self.split_method = self.split_params.splitMethod

    def load_model(self) -> None:

        self.model = load_sbert_cross_encoder_model(model_name=self.model_name, 
                            device=self.device, max_length=self.max_length)['model']

    def explode_nested_content_field(self, inputs_df: pd.DataFrame) -> pd.DataFrame:
        """this is used to chunk the text content and then create a new entry for the model
        based on the chunked content,
        e.g. ['hello. this is a sentence. here is another.']
        if we split the text by sentence then we get
        e.g. ['hello.', 'this is a sentence.', 'here is another.']
        so now we need 3 rows when before we had 1. this function performs the proper
        remapping after splitting/chunking

        Args:
            inputs_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        # used to allow chunking on text in the same way the inexing does it
        _func = partial(text_processor.split_text, split_length=self.split_length, split_overlap=self.split_overlap, split_by=self.split_method)
        inputs_df = inputs_df.merge(inputs_df[Columns.field_content].apply(_func).explode(), left_index=True, right_index=True)
        inputs_df[Columns.field_content_original] = inputs_df[Columns.field_content + '_x']
        inputs_df[Columns.field_content] = inputs_df[Columns.field_content + '_y']
        del inputs_df[Columns.field_content + '_x']
        del inputs_df[Columns.field_content + '_y']
        
        return inputs_df

    def rerank(self, query: str, results: Dict, searchable_attributes: List[str] = None) -> None:
        """the main reranking method

        Args:
            query (str): _description_
            results (Dict): _description_
            searchable_attributes (List[str], optional): _description_. Defaults to None.

        Raises:
            TypeError: _description_
            RuntimeError: _description_
        """
        self.results = results
        self.searchable_attributes = searchable_attributes

        if not isinstance(results, (dict, defaultdict)):
            raise TypeError(f"expected a dict or defaultdict, received {type(results)}")

        if len(results[ResultsFields.hits]) == 0:
            logger.warning("empty results for re-ranking. returning doing nothing...")
            return 

        if self.model is None:
            self.load_model()

        self.format_results(results)

        # we need to create the pairs of data to score over
        # if searchable attributes is None, then we do all non _ fields, 
        # otherwise search over the searchable attributes
        if self.searchable_attributes is None:
            self.searchable_attributes = self.formatted_results.searchable_fields

        # first stage of formatting converts results dict to dataframe
        self.inputs_df = self.formatted_results.format_for_model(self.formatted_results.results_df, self.searchable_attributes, query=query)
        
        # second stage (optionally) add more rows by splitting the content into sub-chunks and
        # performing the apprpriate filling in of other values
        if self.split_params is not None:
            _n = len(self.inputs_df)
            self.inputs_df = self.explode_nested_content_field(self.inputs_df)
            logger.info(f"chunking field content, went from length {_n} to {len(self.inputs_df)}")
        
        # final stage creates list of lists of strings to go straight to the model
        self.model_inputs = self._prepare_inputs(self.inputs_df)

        if not _verify_model_inputs(self.model_inputs):
            raise RuntimeError(f"incorrect model inputs, expected list of lists but recevied {type(self.model_inputs)} and {type(self.model_inputs[0])}")

        self.scores = self.model.predict(self.model_inputs)
        self.scores = _convert_cross_encoder_output(self.scores)

        self.inputs_df[ResultsFields.reranker_score] = self.scores
        self.inputs_df[ResultsFields.hybrid_score_multiply] = np.clip(self.inputs_df[ResultsFields.original_score], 1e-3, np.inf)*np.clip(self.inputs_df[ResultsFields.reranker_score], 1e-3, np.inf)
        self.inputs_df[ResultsFields.hybrid_score_add] = self.inputs_df[ResultsFields.original_score] + self.inputs_df[ResultsFields.reranker_score]

        self.get_reranked_results()


class ReRankerOwl(ReRanker):
    
    """reranker for owl based image reranking
    """

    def __init__(self, model_name: str, device: str, image_size: Tuple):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.image_size = image_size
        self.results = []
    
        self.model = None
        self.processed_inputs = None

        self.results = None
        self.image_attributes = None
        self.num_highlights = None

        self._model_map = None
        self._get_model_mapping()

        if self.model_name not in self._model_map:
            raise RerankerNameError(f"could not find model_name={self.model_name} in mappings {list(self._model_map.keys())}")

    def _get_model_mapping(self):

        self._model_map = {
            "google/owlvit-base-patch32":"google/owlvit-base-patch32",
            "google/owlvit-base-patch16":"google/owlvit-base-patch16",
            "google/owlvit-large-patch14":"google/owlvit-large-patch14",
            "owl/ViT-B/32":"google/owlvit-base-patch32",
            "owl/ViT-B/16":"google/owlvit-base-patch16",
            "owl/ViT-L/14":"google/owlvit-large-patch14",
        }

    def load_model(self):
        
        self._remapped_name = self._model_map[self.model_name]
        logger.info(f"loading model={self._remapped_name} from input name={self.model_name} to device {self.device}")
        loaded = load_owl_vit(self._remapped_name, device=self.device)
        self.model = loaded['model']
        self.processor = loaded['processor']

    @staticmethod
    def load_images(content: List[str], size: Tuple[int]) -> Tuple[ImageType, List[Tuple]]:

        # uses same underlying loader as all other image models ffrom clip_utils
        images, original_size = zip(*[_load_image(f, size=size) for f in content])

        # use highlights
        # get parent value to load from highlight key name
        # only allow rerank if single attribute
        return images, original_size

    def rerank(self, query: str, results: Dict, image_attributes: List, num_highlights: int = 1):
        
        # TODO add image based reranking when it is available
        # https://github.com/huggingface/transformers/pull/20136
        self.results = results
        self.image_attributes = image_attributes
        self.num_highlights = num_highlights

        if not isinstance(results, (dict, defaultdict)):
            raise TypeError(f"expected a dict or defaultdict, received {type(results)}")

        if len(results[ResultsFields.hits]) == 0:
            logger.warning("empty results for re-ranking. returning doing nothing...")
            return 

        if self.model is None:
            self.load_model()

        self.format_results(results, searchable_fields=image_attributes)

        # first stage of formatting converts results dict to dataframe
        self.inputs_df = self.formatted_results.format_for_model(self.formatted_results.results_df, self.image_attributes, query=query)
        
        # final stage creates list of lists of strings to go straight to the model
        self.model_inputs = self._prepare_inputs(self.inputs_df)

        # todo unzip and get image location
        queries, image_names = zip(*self.model_inputs)

        # try:
        self.images, self.original_sizes = self.load_images(image_names, self.image_size)
        
        # # TODO check query type before making a list
        _b, _s, _i, _bo = [], [], [], []

        # # TODO find out why batching images does not work 
        for image, content, _query, orig_size in zip(self.images, image_names, queries, self.original_sizes):
            self.processed_inputs = _process_owl_inputs(self.processor, _query, image).to(self.device)
            owl_results = _predict_owl(self.model, self.processed_inputs, 
                                post_process_function=self.processor.post_process,
                        size=self.image_size)

            boxes, scores, identifier = _process_owl_result(owl_results, content)
            boxes, scores, identifier = sort_owl_boxes_scores(boxes, scores, identifier)
            boxes, scores, identifier = _keep_top_k(boxes, k=num_highlights), _keep_top_k(scores, k=num_highlights), _keep_top_k(identifier, k=num_highlights)

            # just keep top from each image
            _bo += [list(image_processor.rescale_box(b.detach().cpu().tolist(), self.image_size, orig_size)) for b in boxes]
            # _os.append()
            _b.append(boxes)
            _s.append(scores)
            _i += identifier

        self.boxes_original = _bo           
        self.boxes = torch.cat(_b)
        self.scores = torch.cat(_s)
        self.identifier = _i
        
        self.boxes_scores_df = pd.DataFrame(zip(self.boxes.detach().cpu().tolist(), self.scores.detach().cpu().tolist(), 
                                            self.identifier, self.boxes_original), 
                                        columns=[Columns.bbox, ResultsFields.reranker_score, Columns.field_content, Columns.bbox_original])

        # now merge back with original - filed_content is the image pointer and can be used as the identifier
        self.inputs_df = self.boxes_scores_df.merge(self.inputs_df, left_on=Columns.field_content, right_on=Columns.field_content)

        self.get_reranked_results(highlight_content_column=Columns.bbox_original)
        

@functools.lru_cache
def _load_image(filename: str, size: Tuple = None) -> ImageType:
    """loads a PIL image with optional resizing

    Args:
        filename (str): _description_
        size (Tuple, optional): _description_. Defaults to None.

    Returns:
        ImageType: _description_
    """
    im = load_image_from_path(filename, {})
    original_size = im.size
    if size is not None:
        im = im.resize(size).convert('RGB')
        
    return im,original_size
