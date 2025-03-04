from marqo.core.models import MarqoTensorQuery


def get_target_hits_and_additional_hits_from_query(query: MarqoTensorQuery):
    if query.ef_search is not None:
        base_target_hits = min(query.limit + query.offset, query.ef_search)
        additional_hits = max(query.ef_search - (query.limit + query.offset), 0)
    else:
        base_target_hits = query.limit + query.offset
        additional_hits = 0

    # Set target hits provided by query, with minimum of base_target_hits
    if query.target_hits is not None:
        target_hits = max(base_target_hits, query.target_hits)
    else:
        target_hits = base_target_hits

    return target_hits, additional_hits