from timeit import default_timer as timer
from typing import Dict, List, Union, Optional

from marqo.core.exceptions import InvalidFieldNameError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models import MarqoIndex
from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.core.models.marqo_index import IndexType
from marqo.core.utils.vector_interpolation import from_interpolation_method, ZeroSumWeightsError, \
    ZeroMagnitudeVectorError
from marqo.exceptions import InvalidArgumentError
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierLists
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
                  exclude_input_documents: bool = True,
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
                  score_modifiers: Optional[ScoreModifierLists] = None
                  ):
        """
        Recommend documents similar to the provided documents.

        Args:
            index_name: Name of the index to search
            documents: A list of document IDs or a dictionary where the keys are document IDs and the values are weights
            tensor_fields: List of tensor fields to use for recommendation (can include text, image, audio, and video fields)
            interpolation_method: Interpolation method to use for combining vectors
            exclude_input_documents: Whether to exclude the input documents from the search results
            result_count: Number of results to return
            offset: Offset of the first result
            highlights: Whether to include highlights in the results
            ef_search: ef_search parameter for HNSW search
            approximate: Whether to use approximate search
            searchable_attributes: List of attributes to search in
            verbose: Verbosity level
            reranker: Reranker to use
            filter: Filter string
            attributes_to_retrieve: List of attributes to retrieve
            score_modifiers: Score modifiers to apply
        """
        # TODO - Extract search and get_docs from tensor_search and refactor this
        # TODO - The dependence on Config in tensor_search is bad design. Refactor to require specific dependencies
        from marqo import config
        from marqo.tensor_search import tensor_search
        from marqo.tensor_search import index_meta_cache

        if documents is None or len(documents) == 0:
            raise InvalidArgumentError('No document IDs provided')

        # remove docs with zero weight
        original_documents = documents
        if isinstance(documents, dict):
            documents = {k: v for k, v in documents.items() if v != 0}
            document_ids = list(documents.keys())
            all_document_ids = list(original_documents.keys())
        else:
            document_ids = documents
            all_document_ids = original_documents

        if len(documents) == 0:
            raise InvalidArgumentError('No documents with non-zero weight provided')

        marqo_index = index_meta_cache.get_index(index_management=self.index_management, index_name=index_name)

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

        t0 = timer()

        marqo_documents = tensor_search.get_documents_by_ids(
            config.Config(self.vespa_client),
            index_name, document_ids, show_vectors=True
        ).dict(exclude_none=True, by_alias=True)

        # Make sure all documents were found
        not_found = []
        for document in marqo_documents['results']:
            if not document['_found']:
                not_found.append(document['_id'])

        if len(not_found) > 0:
            raise InvalidArgumentError(f'The following document IDs were not found: {", ".join(not_found)}')

        doc_vectors: Dict[str, List[List[float]]] = {}
        docs_without_vectors = []
        for document in marqo_documents['results']:
            vectors: List[List[float]] = []
            for tensor_facet in document['_tensor_facets']:
                field = list(tensor_facet.keys())[0]
                if tensor_fields is None or field in tensor_fields:
                    vectors.append(tensor_facet['_embedding'])

            doc_vectors[document['_id']] = vectors

            if len(vectors) == 0:
                docs_without_vectors.append(document['_id'])

        if len(docs_without_vectors) > 0:
            raise InvalidArgumentError(
                f'The following documents do not have embeddings: {", ".join(docs_without_vectors)}'
            )

        vectors: List[List[float]] = []
        weights: List[float] = []

        for document_id, vector_list in doc_vectors.items():
            if isinstance(documents, dict):
                weight = documents[document_id]
            else:
                weight = 1
            vectors.extend(vector_list)
            weights.extend([weight] * len(vector_list))

        try:
            interpolated_vector = vector_interpolation.interpolate(
                vectors, weights
            )
        except ZeroSumWeightsError as e:
            if interpolation_method == InterpolationMethod.SLERP:
                raise InvalidArgumentError(
                    'Sum of one or more consecutive weights is zero. '
                    'SLERP cannot interpolate vectors with zero sum of weights. Such weight pairs are prone to causing '
                    'this error depending on document embeddings, and should be avoided',
                    cause=e
                ) from e
            else:  # lerp or nlerp
                raise InvalidArgumentError(
                    'Sum of weights is zero. LERP/NLERP requires non-zero sum of weights',
                    cause=e
                ) from e
        except ZeroMagnitudeVectorError as e:
            if interpolation_method == InterpolationMethod.NLERP:
                raise InvalidArgumentError(
                    'Linear interpolation of embeddings led to a zero-magnitude vector. '
                    'NLERP cannot normalize a vector with zero magnitude',
                    cause=e
                ) from e
            else:  # shouldn't reach here
                raise e

        if exclude_input_documents:
            # Make sure to include zero-weight documents in this filter
            recommend_filter = self._get_exclusion_filter(marqo_index, all_document_ids, filter)
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
            score_modifiers=score_modifiers,
            processing_start=t0
        )

        return results

    def _get_default_interpolation_method(self, marqo_index: MarqoIndex) -> InterpolationMethod:
        if marqo_index.normalize_embeddings:
            return InterpolationMethod.SLERP
        else:
            return InterpolationMethod.LERP

    def _get_exclusion_filter(self, marqo_index: MarqoIndex, documents: List[str], user_filter: Optional[str]) -> str:
        if marqo_index.type == IndexType.Structured:
            not_in = 'NOT _id IN (' + ', '.join([f'{doc}' for doc in documents]) + ')'
        else:
            not_in = 'NOT (' + ' OR '.join([f'_id:({doc})' for doc in documents]) + ')'

        if user_filter is not None and user_filter.strip() != '':
            return f'({user_filter}) AND {not_in}'
        else:
            return not_in
