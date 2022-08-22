from functools import lru_cache
import requests
import validators
import time
import uuid
from collections import defaultdict

import pandas as pd
from PIL import Image
import numpy as np

from marqo.s2_inference.types import *
from marqo.s2_inference.reranking.model_utils import (
    load_sbert_cross_encoder_model,
    load_owl_vit,
    _process_owl_inputs,
    _predict_owl,
    process_owl_results,
    sort_owl_boxes_scores
)

from marqo.s2_inference.logger import get_logger
logger = get_logger(__name__)

# TODO move this to an enum
HIGHLIGHTS_FIELD = '_highlights'



def _get_ids_from_results(results):

    return [doc['_id'] for doc in results['hits']]

def get_results_by_doc_id(results):

    ids = _get_ids_from_results(results)

class FormattedResults:
    
    # helper class to format results
    # use this as the interface between results and models and reranking

    # output should be a dataframe with all fields required 

    def __init__(self, results, highlights_field=HIGHLIGHTS_FIELD):

        self.results = results
        self.highlights_field = highlights_field

        # check ids exist and if not create some
        self._fill_doc_ids(results)

        self.results_to_df()

        self._get_searchable_columns()

    def results_to_df(self):
        self.results_df = pd.DataFrame(self.results['hits'])

        def _get_highlights(content):
            if content == []:
                return None
            elif isinstance(content, (dict, defaultdict)):
                _key = list(content.keys())
                if len(_key) == 0:
                    return None
                if len(_key) > 1:
                    logger.warning(f"found more than 1 highlight. found {_key}. using the first only for re-ranking...")
                return content[_key[0]]

        if self.highlights_field in self.results_df:
            self.results_df[self.highlights_field] = self.results_df[self.highlights_field].apply(_get_highlights)


    @staticmethod
    def _fill_doc_ids(results):
        # check if an id exists, otherwise create a temporary one for easier identification
        # during re-ranking
        
        for result in results['hits']:
            if '_id' not in result:
                doc_id = str(uuid.uuid4())
                result['_rerank_id'] = doc_id 
            else:
                result['_rerank_id'] = result['_id']

    def _get_searchable_columns(self):
        self.searchable_fields = [field for field in self.results_df.columns.tolist() if not field.startswith('_')]

    @staticmethod
    def format_for_model(results_df, searchable_fields, query=None):
        
        # we want the output here to be tuples of the attribute content,query, id and attribute that was used
        # the first two will go to the model while the latter will be used to report the results
        inputs = []
        for field in searchable_fields:
            _inputs_df = results_df[[field, '_rerank_id', '_score']]
            _inputs_df['field_name'] = field

            # this is the case if we have some documents that do not have all the fields, 
            # the empty fields go to na when converted to pandas
            _inputs_df = _inputs_df[_inputs_df[field].notna()] 

            if len(_inputs_df) > 0:
                
                _inputs_df['query'] = query

                # we keep it in case we want to do hybrid search
                if '_score' not in _inputs_df:
                    _inputs_df['_score'] = 1.0

                fields_to_keep = ['query', field, '_rerank_id', 'field_name', '_score']

                inputs += _inputs_df[fields_to_keep].values.tolist()
            
        inputs_df = pd.DataFrame(inputs, columns=['query', 'field_content', '_rerank_id', 'original_field_name',  '_score'])
        
        return inputs_df


    # def get_results_mapped_by_id(self):

    #     for result in self.results['hits']:
    #         doc_id = result['_rerank_id']
    #         self.results_mapped_by_id[doc_id].append(result)

    # def get_attributes(self):
    #     # get the attributes of the documents
    #     for result in self.results['hits']:
    #         self.attributes = 1


    # def get_highlighted_content(self):
    #     # used when we have a non-empty highlights field
    #     if len(self.results['hits']) == 0:
    #         self.highlighted_content = None
    #         return 

    #     if self.highlights_field not in self.results['hits'][0]:
    #         raise KeyError(f'could not find {self.highlights_field} field in results')

    #     # we need to return if highlights are empty. we also assume that if the first record has no highlights, 
    #     # then none do
    #     if len(self.results['hits'][0][self.highlights_field]) == 0:
    #         self.highlighted_content = None
    #         return 

    #     highlighted_keys = list(self.results['hits'][0][self.highlights_field].keys())
    #     # should only be 1 highlight at the moment
    #     if len(highlighted_keys) != 1:
    #         raise RuntimeError(f"too many highlights, expected 1 but received {len(highlighted_keys)}")
    #     self.highlighted_field = highlighted_keys[0]

    #     # now get the dat ready for the re-ranker
    #     self.highlighted_content = [data[self.highlighted_field] for data in self.results['hits']]
    #     self.doc_ids = [data['_id'] for data in self.results['hits']]

    #     # we want to output the content so it can be passed to a reranker
    #     self.rerank_over_highlights_inputs = pd.DataFrame()

    # # def get_all_content(self):
    # #     # used to get content, and ids for passing to the model

    # def _process(self):
class ReRanker:

    def __init__(self):
        pass 


    def load_model(self):
        pass

    
    def format_results(self, results, query=None):
        self.results = results
        self.formatted_results = FormattedResults(results)
        


    def rerank(self, query, results):
        pass

# TODO add generic HF models
class ReRankerText(ReRanker):

    def __init__(self, model_name: str, device: str = 'cpu', max_length: int = 512, num_highlights=1):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.num_highlights = num_highlights
        self.formatted_results = None

        self.model = None

    def load_model(self):

        self.model = load_sbert_cross_encoder_model(self.model_name, self.max_length)['model']

    @staticmethod
    def _prepare_inputs(inputs_df, query_column='query', content_column='field_content'):
        # TODO add in chunking option
        return inputs_df[[query_column, content_column]].values.tolist()

    def rerank(self, query, results, searchable_attributes=None):

        self.results = results
        self.searchable_attributes = searchable_attributes

        if not isinstance(results, (dict, defaultdict)):
            raise TypeError(f"expected a dict or defaultdict, received {type(results)}")

        if len(results['hits']) == 0:
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

        # now loop through each combination of query and searchable fields - could also just smash everything together here as well
        self.inputs_df = self.formatted_results.format_for_model(self.formatted_results.results_df, self.searchable_attributes, query=query)
        # TODO chunk the text and send to model
        self.model_inputs = self._prepare_inputs(self.inputs_df)

        self.scores = self.model.predict(self.model_inputs)

        self.inputs_df['_score_rerank'] = self.scores
        self.inputs_df['_score_multiply'] = np.clip(self.inputs_df['_score'], 1e-3, np.inf)*np.clip(self.inputs_df['_score_rerank'], 1e-3, np.inf)
        self.inputs_df['_score_add'] = self.inputs_df['_score'] + self.inputs_df['_score_rerank']

        self.get_reranked_results()

    def get_reranked_results(self, score_column='_score_rerank'):
        
        reranked_top = []
        for i,group in self.inputs_df.groupby('_rerank_id'):
            group = group.sort_values(score_column, ascending=False).head(self.num_highlights)
            reranked_top.append(group)
        reranked_top = pd.concat(reranked_top).set_index('_rerank_id')
        
        # find out which document is the highest, then do the field as a highlight
        for result in self.results['hits']:
            rerank_id = result['_rerank_id']
            if self.num_highlights == 1:
                _score = reranked_top.loc[rerank_id][score_column]
            else:
                _score = reranked_top.loc[rerank_id][score_column].values.tolist()
                
            result['_score_rerank'] = _score
            _df = reranked_top.loc[rerank_id]
            #import ipdb;ipdb.set_trace()
            if self.num_highlights == 1:
                _content = [{_df['original_field_name']:_df['field_content']}]
            else:
                _content = [{row['original_field_name']:row['field_content']} for _,row in _df.iterrows()]

            result['_highlights_rerank'] = _content

        self.results['hits'] = sorted(self.results['hits'], key=lambda x:x['_score_rerank'], reverse=True)

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
        self.content, self.ids = self.get_data(results, highlights_field=HIGHLIGHTS_FIELD)
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
        #import ipdb;ipdb.set_trace()
        self.boxes, self.scores, self.identifier = sort_owl_boxes_scores(boxes, scores, self.identifier)
        t_sort = time.time()

        timings = {"time_to_prepare_data":t_image_loaded - t_start, 'time_to_predict':t_predict-t_image_loaded, 'time_to_sort':t_sort - t_predict}

        # do you take 1 or n from each image?
        # TODO prepatch if we want 
        # TODO take top n from each one
        results['reranked'] = {}
        results['reranked']['hits'] = []
        results['reranked']['hits'] = [{'boxes':self.boxes, 'scores':self.scores, 'identifier':self.identifier}]
        results['reranked']['processTime'] = timings

        return results

def rerank_search_results(search_result, query, model_name, device, 
                searchable_attributes=None, num_highlights=1):

    # TODO add in routing and move this into another file for interfacing
    reranker = ReRankerText(model_name, device=device, num_highlights=num_highlights)
    reranker.rerank(query=query, results=search_result, searchable_attributes=searchable_attributes)



def _load_image(filename, size=None):
    
    is_url = validators.url(filename)
    print(filename, is_url)
    if is_url:
        im = Image.open(requests.get(filename, stream=True).raw)
    else:
        im = Image.open(filename)    
    #import ipdb;ipdb.set_trace()
    if size is None:
        size = im.size
    #im.draft('RGB', size)
    im = im.resize(size).convert('RGB')
    return im


if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    texts = [["a photo of a cat", "a photo of a dog"]]


    # i = 0  # Retrieve predictions for the first image for the corresponding text queries
    # text = texts[i]
    # boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # score_threshold = 0.1
    # for box, score, label in zip(boxes, scores, labels):
    #     box = [round(i, 2) for i in box.tolist()]
    #     if score >= score_threshold:
    #         print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

    # @lru_cache(maxsize=None)
    # def _load_image(filenmae):
    #     pass

    # def load_images():
    #     # https://stackoverflow.com/questions/57663734/how-to-speed-up-image-loading-in-pillow-python
    #     pass
