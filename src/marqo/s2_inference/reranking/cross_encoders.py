from functools import partial
import requests
import validators
import time
import uuid
from collections import defaultdict

import pandas as pd
pd.options.mode.chained_assignment = None
from PIL import Image
import numpy as np
import torch

from marqo.s2_inference.types import *
from marqo.s2_inference.reranking.model_utils import (
    load_sbert_cross_encoder_model,
    load_owl_vit,
    _process_owl_inputs,
    _predict_owl,
    process_owl_results,
    sort_owl_boxes_scores,
    _verify_model_inputs
)
from marqo.s2_inference.reranking.enums import Columns, ResultsFields
from marqo.s2_inference.reranking.configs import get_default_text_processing_parameters
from marqo.processing import text as text_processor

from marqo.s2_inference.logger import get_logger
logger = get_logger(__name__)

def _get_ids_from_results(results):

    return [doc[ResultsFields.id] for doc in results[ResultsFields.hits]]

def get_results_by_doc_id(results):

    ids = _get_ids_from_results(results)

class FormattedResults:

    """
    helper class to format results.
    use this as the interface between results and models and reranking.
    output should be a dataframe with all fields required.
    """    
    
    def __init__(self, results: Dict, highlights_field: str = ResultsFields.highlights):

        self.results = results
        self.highlights_field = highlights_field
        
        # check ids exist and if not create some
        self._fill_doc_ids(self.results)

        self.results_to_df()

        self._get_searchable_columns()

    def results_to_df(self) -> None:
        """_summary_

        Returns:
            _type_: _description_
        """
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

    def format_results(self, results: Dict, query: str = None):
        """standardize the way the results are formatted to go to a standard cross-encoder

        Args:
            results (Dict): _description_
            query (str, optional): _description_. Defaults to None.
        """
        self.results = results
        self.formatted_results = FormattedResults(self.results)

    def rerank(self, query, results):
        pass

class ReRankerText(ReRanker):

    def __init__(self, model_name: str, device: str = 'cpu', max_length: int = 512, num_highlights: int = 1, 
                        split_params: Dict = get_default_text_processing_parameters()):
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
        self.split_length = self.split_params['split_length']
        self.split_overlap = self.split_params['split_overlap']
        self.split_method = self.split_params['split_method']

    def load_model(self) -> None:

        self.model = load_sbert_cross_encoder_model(model_name=self.model_name, 
                            device=self.device, max_length=self.max_length)['model']

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

        self.inputs_df[ResultsFields.reranker_score] = self.scores
        self.inputs_df[ResultsFields.hybrid_score_multiply] = np.clip(self.inputs_df[ResultsFields.original_score], 1e-3, np.inf)*np.clip(self.inputs_df[ResultsFields.reranker_score], 1e-3, np.inf)
        self.inputs_df[ResultsFields.hybrid_score_add] = self.inputs_df[ResultsFields.original_score] + self.inputs_df[ResultsFields.reranker_score]

        self.get_reranked_results()

    def get_reranked_results(self, score_column: str = ResultsFields.reranker_score):
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
                _content = {_df[Columns.original_field_name]:_df[Columns.field_content]}
            else:
                _content = [{row[Columns.original_field_name]:row[Columns.field_content]} for _,row in _df.iterrows()]

            result[ResultsFields.highlights_reranked] = _content

        self.results[ResultsFields.hits] = sorted(self.results[ResultsFields.hits], key=lambda x:x[ResultsFields.reranker_score], reverse=True)


class ReRankerOwl(ReRanker):
# we might need the index config to get the processing params

    def __init__(self, model_name: str, device: str, image_size: Tuple):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.image_size = image_size
        self.results = []
    
    def load_model(self):
        
        loaded = load_owl_vit(self.device)
        self.model = loaded['model']
        self.processor = loaded['processor']

    @staticmethod
    def load_images(content, size):

        # TODO do web urls as well - fast laoding could be hard -
        images = [_load_image(f, size=size) for f in content]

        # use highlights
        # get parent value to load from highlight key name
        # only allow rerank if single attribute
        return images

    def rerank(self, query, results):
        t_start = time.time()
        self.content, self.ids = self.get_data(results, highlights_field=ResultsFields.highlights)
        t_content = time.time()
        self.images = self.load_images(self.content, self.image_size)
        t_image_loaded = time.time()
        # TODO check query type before making a list
        _b, _s, _i, _im = [], [], [], []

        # TODO find out why batching images does not work 
        for image, content in zip(self.images, self.content):
            self.processed_inputs = _process_owl_inputs(self.processor, [[query]], image)
            owl_results = _predict_owl(self.model, self.processed_inputs, 
                                post_process_function=self.processor.post_process,
                        size=self.image_size)

            boxes, scores, identifier = _process_owl_result(owl_results, content)
            _b.append(boxes)
            _s.append(scores)
            _i += identifier
            _im.append(image)
        t_predict = time.time()
        boxes = torch.cat(_b)
        scores = torch.cat(_s)
        self.identifier = _i
        images = _im
         
        self.boxes, self.scores, self.identifier = sort_owl_boxes_scores(boxes, scores, self.identifier)
        t_sort = time.time()

        timings = {"time_to_prepare_data":t_image_loaded - t_start, 'time_to_predict':t_predict-t_image_loaded, 'time_to_sort':t_sort - t_predict}

        # do you take 1 or n from each image?
        # TODO prepatch if we want 
        # TODO take top n from each one
        results['reranked'] = {}
        results['reranked'][ResultsFields.hits] = []
        results['reranked'][ResultsFields.hits] = [{'boxes':self.boxes, 'scores':self.scores, 'identifier':self.identifier}]
        results['reranked']['processTime'] = timings

        return results

def _load_image(filename: str, size: Tuple = None) -> ImageType:
    """loads a PIL image

    Args:
        filename (str): _description_
        size (Tuple, optional): _description_. Defaults to None.

    Returns:
        ImageType: _description_
    """
    is_url = validators.url(filename)
    print(filename, is_url)
    if is_url:
        im = Image.open(requests.get(filename, stream=True).raw)
    else:
        im = Image.open(filename)    
     
    if size is None:
        size = im.size
    #im.draft('RGB', size)
    im = im.resize(size).convert('RGB')
    return im
