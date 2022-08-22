class Columns:
    query = 'query'
    field_content = 'field_content'
    original_field_name = 'original_field_name'

class ResultsFields:
    hits = 'hits'
    original_score = '_score'
    reranker_score = '_score_rerank'    
    hybrid_score_multiply = '_score_multiply'
    hybrid_score_add = '_score_add'
    highlights = '_highlights'
    reranked_id = '_rerank_id'
    highlights_reranked = '_highlights_reranked'
