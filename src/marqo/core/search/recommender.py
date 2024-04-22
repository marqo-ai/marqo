from typing import Dict, List, Union, Optional

from marqo import config
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models import MarqoIndex
from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.core.utils.vector_interpolation import from_interpolation_method
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
                  searchable_attributes: Optional[List[str]] = None,
                  filter: str = None,
                  exclude_input_documents=True,
                  interpolation_method: Optional[InterpolationMethod] = None
                  ):
        # TODO - Extract search and get_docs from tensor_search and refactor this
        # TODO - The dependence on Config in tensor_search is bad design. Refactor to require specific dependencies
        from marqo.tensor_search import tensor_search
        from marqo.tensor_search import index_meta_cache

        if interpolation_method is None:
            marqo_index = index_meta_cache.get_index(config.Config(self.vespa_client), index_name=index_name)
            interpolation_method = self._get_default_interpolation_method(marqo_index)

        vector_interpolation = from_interpolation_method(interpolation_method)

        if isinstance(documents, dict):
            document_ids = list(documents.keys())
        else:
            document_ids = documents

        marqo_documents = tensor_search.get_documents_by_ids(
            config.Config(self.vespa_client),
            index_name, document_ids, show_vectors=True
        )

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

        interpolated_vector = vector_interpolation.interpolate(
            vectors, weights
        )

        if exclude_input_documents:
            filter = self._get_exclusion_filter(document_ids, filter)

        results = tensor_search.search(
            config.Config(self.vespa_client),
            index_name,
            text=None,
            context=SearchContext(tensor=[SearchContextTensor(vector=interpolated_vector, weight=1)]),
            searchable_attributes=searchable_attributes,
            filter=filter
        )

        return results

    def _get_default_interpolation_method(self, marqo_index: MarqoIndex) -> InterpolationMethod:
        if marqo_index.normalize_embeddings:
            return InterpolationMethod.SLERP
        else:
            return InterpolationMethod.LERP

    def _get_exclusion_filter(self, documents: List[str], user_filter: Optional[str]) -> str:
        not_in = 'NOT _id IN ("' + '","'.join(documents) + '")'

        if user_filter is not None and user_filter.strip() != '':
            return f'({user_filter}) AND {not_in}'
        else:
            return not_in
