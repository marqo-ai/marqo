from marqo.core.models import MarqoTensorQuery


def get_rerank_depth_and_additional_hits_from_query(query: MarqoTensorQuery):
    if query.ef_search is not None:
        base_rerank_depth = min(query.limit + query.offset, query.ef_search)
        additional_hits = max(query.ef_search - (query.limit + query.offset), 0)
    else:
        base_rerank_depth = query.limit + query.offset
        additional_hits = 0

    # Set rerank depth provided by query, with minimum of base_rerank_depth
    if query.rerank_depth is not None:
        rerank_depth = max(base_rerank_depth, query.rerank_depth)
    else:
        rerank_depth = base_rerank_depth

    return rerank_depth, additional_hits