from timeit import default_timer as timer
from typing import Dict, List, Union, Optional

from marqo.core.exceptions import InvalidFieldNameError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models import MarqoIndex
from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.core.models.marqo_index import IndexType
from marqo.core.utils.vector_interpolation import from_interpolation_method
from marqo.exceptions import InvalidArgumentError
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.score_modifiers_object import ScoreModifier
from marqo.tensor_search.models.search import SearchContext, SearchContextTensor
from marqo.vespa.vespa_client import VespaClient


class Recommender:
    def __init__(self, vespa_client: VespaClient, index_management: IndexManagement):
        self.vespa_client = vespa_client
        self.index_management = index_management

    def recommend(self,
                  index_name: str,
                  documents: Union[List[str], Dict[str, float]],
                  tensor_fields: Optional[List[str]] = None,
                  interpolation_method: Optional[InterpolationMethod] = None,
                  exclude_input_documents=True,
                  result_count: int = 3,
                  offset: int = 0,
                  highlights: bool = True,
                  ef_search: Optional[int] = None,
                  approximate: Optional[bool] = None,
                  searchable_attributes: Optional[List[str]] = None,
                  verbose: int = 0,
                  reranker: Union[str, Dict] = None,
                  filter: str = None,
                  attributes_to_retrieve: Optional[List[str]] = None,
                  device: str = None,
                  score_modifiers: Optional[ScoreModifier] = None,
                  model_auth: Optional[ModelAuth] = None
                  ):
        # TODO - Extract search and get_docs from tensor_search and refactor this
        # TODO - The dependence on Config in tensor_search is bad design. Refactor to require specific dependencies
        from marqo import config
        from marqo.tensor_search import tensor_search
        from marqo.tensor_search import index_meta_cache

        if documents is None or len(documents) == 0:
            raise InvalidArgumentError('No document IDs provided')

        marqo_index = index_meta_cache.get_index(config.Config(self.vespa_client), index_name=index_name)

        if interpolation_method is None:
            interpolation_method = self._get_default_interpolation_method(marqo_index)

        vector_interpolation = from_interpolation_method(interpolation_method)

        if marqo_index.type == IndexType.Structured:
            # Validate tensor field names
            if tensor_fields is not None:
                valid_tensor_fields = marqo_index.tensor_field_map.keys()
                for tensor_field in tensor_fields:
                    if tensor_field not in valid_tensor_fields:
                        raise InvalidFieldNameError(f'Tensor field "{tensor_field}" not found in index "{index_name}". '
                                         f'Available tensor fields: {", ".join(valid_tensor_fields)}')

        if isinstance(documents, dict):
            document_ids = list(documents.keys())
        else:
            document_ids = documents

        t0 = timer()

        marqo_documents = tensor_search.get_documents_by_ids(
            config.Config(self.vespa_client),
            index_name, document_ids, show_vectors=True
        )

        # Make sure all documents were found
        not_found = []
        for document in marqo_documents['results']:
            if not document['_found']:
                not_found.append(document['_id'])

        if len(not_found) > 0:
            raise InvalidArgumentError(f'The following document IDs were not found: {", ".join(not_found)}')

        doc_vectors: Dict[str, List[List[float]]] = {}
        for document in marqo_documents['results']:
            vectors: List[List[float]] = []
            for tensor_facet in document['_tensor_facets']:
                field = list(tensor_facet.keys())[0]
                if tensor_fields is None or field in tensor_fields:
                    vectors.append(tensor_facet['_embedding'])

            doc_vectors[document['_id']] = vectors

        vectors: List[List[float]] = []
        weights: List[float] = []

        for document_id, vector_list in doc_vectors.items():
            if isinstance(documents, dict):
                weight = documents[document_id]
            else:
                weight = 1
            vectors.extend(vector_list)
            weights.extend([weight] * len(vector_list))

        if len(vectors) == 0:
            time_taken = timer() - t0
            return {
                'query': None,
                'hits': [],
                'limit': result_count,
                'offset': offset,
                'processingTimeMs': round(time_taken * 1000)
            }

        interpolated_vector = vector_interpolation.interpolate(
            vectors, weights
        )

        if exclude_input_documents:
            recommend_filter = self._get_exclusion_filter(document_ids, filter)
        else:
            recommend_filter = filter

        results = tensor_search.search(
            config.Config(self.vespa_client),
            index_name,
            text=None,
            context=SearchContext(tensor=[SearchContextTensor(vector=interpolated_vector, weight=1)]),
            result_count=result_count,
            offset=offset,
            highlights=highlights,
            ef_search=ef_search,
            approximate=approximate,
            searchable_attributes=searchable_attributes,
            verbose=verbose,
            reranker=reranker,
            filter=recommend_filter,
            attributes_to_retrieve=attributes_to_retrieve,
            device=device,
            score_modifiers=score_modifiers,
            model_auth=model_auth,
            processing_start=t0
        )

        return results

    def _get_default_interpolation_method(self, marqo_index: MarqoIndex) -> InterpolationMethod:
        if marqo_index.normalize_embeddings:
            return InterpolationMethod.SLERP
        else:
            return InterpolationMethod.LERP

    def _get_exclusion_filter(self, documents: List[str], user_filter: Optional[str]) -> str:
        not_in = 'NOT (' + ' OR '.join([f'_id:({doc})' for doc in documents]) + ')'

        if user_filter is not None and user_filter.strip() != '':
            return f'({user_filter}) AND {not_in}'
        else:
            return not_in
