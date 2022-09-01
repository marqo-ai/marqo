# use this as the entry point for reranking
from marqo.s2_inference.reranking.enums import ResultsFields
from marqo.s2_inference.reranking.cross_encoders import ReRankerText, ReRankerOwl
from marqo.s2_inference.types import Dict, List

def rerank_search_results(search_result: Dict, query: str, model_name: str, device: str, 
                searchable_attributes: List[str] = None, num_highlights: int = 1, 
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

    # TODO add in routing based on model type
    if 'owl' in model_name.lower():
        reranker = ReRankerOwl(model_name=model_name, device=device, image_size=(240,240))
        reranker.rerank(query=query, results=search_result, image_attributes=searchable_attributes)
    else:
        reranker = ReRankerText(model_name=model_name, device=device, num_highlights=num_highlights)
        reranker.rerank(query=query, results=search_result, searchable_attributes=searchable_attributes)

    if overwrite_original_scores_highlights:
        cleanup_final_reranked_results(search_result)

def cleanup_final_reranked_results(reranked_results: Dict) -> None:
    """removes the fields that were created for the reranking process

    Args:
        reranked_results (Dict[List]): _description_
    """
    for result in reranked_results['hits']:
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
