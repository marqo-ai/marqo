# use this as the entry point for reranking
from PIL import UnidentifiedImageError

from marqo.s2_inference.errors import RerankerError, RerankerNameError
from marqo.s2_inference.logger import get_logger
from marqo.s2_inference.reranking.cross_encoders import ReRankerText, ReRankerOwl
from marqo.s2_inference.reranking.enums import ResultsFields
from marqo.s2_inference.types import Dict, List, Optional

logger = get_logger(__name__)

def rerank_search_results(search_result: Dict, query: str, model_name: str, device: str, 
                searchable_attributes: Optional[List[str]] = None, num_highlights: int = 1,
                overwrite_original_scores_highlights: bool = True) -> None:
    """the parent function to handle calling the rerankers. the results are modified in place

    Args:
        search_result (Dict[List]): _description_
        query (str): _description_
        model_name (str): _description_
        device (str): _description_
        searchable_attributes (List[str], optional): _description_. Defaults to None.
        num_highlights (int, optional): _description_. Defaults to 1.
        overwrite_original_scores_highlights (bool, optional): _description_. Defaults to True.
    """

    # check the search_results have the searchable attribute before proceeding
    # skip reranking if the results do not contain the field
    if not _check_searchable_fields_in_results(search_results=search_result, searchable_fields=searchable_attributes):
        return search_result

    if 'owl' in model_name.lower():
        # owl needs the image location, while the text based ones can handle different number of fields but concat the text
        if searchable_attributes in (None, [], (), ''):
            raise RerankerError(f"found searchable_attributes={searchable_attributes} but expected list of strings for {model_name}")
        
        if len(searchable_attributes) > 1:
            logger.info(f"currently only a single attribute can be reranked over for {model_name}. taking the first field {[searchable_attributes[0]]} from {searchable_attributes}")
            searchable_attributes = [searchable_attributes[0]]

        try:
            reranker = ReRankerOwl(model_name=model_name, device=device, image_size=(240,240))
            reranker.rerank(query=query, results=search_result, image_attributes=searchable_attributes)
        except (UnidentifiedImageError, RerankerNameError) as e:
            raise RerankerError(message=str(e)) from e

    else:
        try:
            reranker = ReRankerText(model_name=model_name, device=device, num_highlights=num_highlights)
            reranker.rerank(query=query, results=search_result, searchable_attributes=searchable_attributes)
        except Exception as e:
            raise RerankerError(message=str(e)) from e

    if overwrite_original_scores_highlights:
        cleanup_final_reranked_results(search_result)

def _check_searchable_fields_in_results(search_results: Dict, searchable_fields: List[str] = None) -> bool:
    """
    checks the searchable fileds are in the search result
    None for searchable fields is allowed and will return True

    Args:
        search_results (Dict): _description_
        searchable_fields (List[str], optional): _description_. Defaults to None.

    Returns:
        bool: _description_
    """

    if searchable_fields == None:
        return True
    return any([True for r in search_results[ResultsFields.hits] if any(s in r for s in searchable_fields)])
    
def cleanup_final_reranked_results(reranked_results: Dict) -> None:
    """removes the fields that were created for the reranking process

    Args:
        reranked_results (Dict[List]): _description_
    """
    for result in reranked_results[ResultsFields.hits]:
        # replace original with reranked score
        # could also do a hybrid score
        if ResultsFields.reranker_score in result:
            result[ResultsFields.original_score] = result[ResultsFields.reranker_score]
            del result[ResultsFields.reranker_score]

        # replace highlights with reranked highlights
        if ResultsFields.highlights_reranked in result:
            result[ResultsFields.highlights] = result[ResultsFields.highlights_reranked]
            del result[ResultsFields.highlights_reranked]
        
        # remove our own internal id from the reranking process
        if ResultsFields.reranked_id in result:
            del result[ResultsFields.reranked_id]
