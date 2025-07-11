from timeit import default_timer as timer
from typing import Dict, List, Union, Optional

from marqo.core.exceptions import InvalidFieldNameError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.inference.api import Inference
from marqo.core.models import MarqoIndex
from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.core.models.marqo_index import IndexType
from marqo.core.utils.vector_interpolation import from_interpolation_method, AllZeroWeightsError, \
    ZeroMagnitudeVectorError
from marqo.exceptions import InvalidArgumentError
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierLists
from marqo.tensor_search.models.search import SearchContext, SearchContextTensor
from marqo.vespa.vespa_client import VespaClient
from marqo.core.unstructured_vespa_index import common as unstructured_common
from marqo.tensor_search import utils, validation


class Recommender:
    def __init__(self, vespa_client: VespaClient, index_management: IndexManagement, inference: Inference):
        self.vespa_client = vespa_client
        self.index_management = index_management
        self.inference = inference

    def get_doc_vectors_from_ids(self,
                  index_name: str,
                  documents: Union[List[str], Dict[str, float]],
                  tensor_fields: Optional[List[str]] = None,
                  concurrency: Optional[int] = None) -> Dict[str, List[List[float]]]:
        """
        This method gets documents from Vespa using their IDs, removes any unnecessary data, checks for
        lack of vectors, then returns a list of document vectors. Can be used internally (in recommend)
        or externally (in the search module).

        Args:
            index_name: Name of the index to search
            documents: A list of document IDs or a dictionary where the keys are document IDs and the values are weights
            tensor_fields: List of tensor fields to use for recommendation (can include text, image, audio, and video fields)
            concurrency: Max number of concurrent requests to use when fetching documents by batch

        Returns:
            A dictionary mapping document IDs to lists of vector embeddings. This is flattened to 1 list per document
                ID (not separated by tensor field). Order of embeddings is not guaranteed.
        """

        # TODO - Extract search and get_docs from tensor_search and refactor this
        from marqo import config
        from marqo.tensor_search import tensor_search, index_meta_cache

        if documents is None or len(documents) == 0:
            raise InvalidArgumentError('No document IDs provided')

        # Check for duplicate document IDs when documents is a list
        if isinstance(documents, list):
            unique_docs = set(documents)
            if len(unique_docs) != len(documents):
                duplicates = [doc for doc in unique_docs if documents.count(doc) > 1]
                raise InvalidArgumentError(f'Duplicate document IDs found: {", ".join(duplicates)}')

        # remove docs with zero weight
        original_documents = documents
        if isinstance(documents, dict):
            documents = {k: v for k, v in documents.items() if v != 0}
            document_ids = list(documents.keys())
            all_document_ids = list(original_documents.keys())
        else:
            document_ids = documents
            all_document_ids = original_documents

        # Validate all IDS
        document_ids = [validation.validate_id(id) for id in document_ids]

        if len(documents) == 0:
            raise InvalidArgumentError('No documents with non-zero weight provided')

        marqo_index = index_meta_cache.get_index(index_management=self.index_management, index_name=index_name)

        if marqo_index.type == IndexType.Structured:
            # Validate tensor field names
            if tensor_fields is not None:
                valid_tensor_fields = marqo_index.tensor_field_map.keys()
                for tensor_field in tensor_fields:
                    if tensor_field not in valid_tensor_fields:
                        raise InvalidFieldNameError(f'Tensor field "{tensor_field}" not found in index "{index_name}". '
                                                    f'Available tensor fields: {", ".join(valid_tensor_fields)}')

        # Use the new optimized method to get only embeddings
        doc_embeddings_by_field = tensor_search.get_doc_vectors_per_tensor_field_by_ids(
            config.Config(self.vespa_client, inference=self.inference),
            index_name, 
            document_ids, 
            tensor_fields=tensor_fields,
            concurrency=concurrency
        )

        # Check that all documents were found
        not_found = []
        for doc_id in document_ids:
            if doc_id not in doc_embeddings_by_field:
                not_found.append(doc_id)

        if len(not_found) > 0:
            raise InvalidArgumentError(f'The following document IDs were not found: {", ".join(not_found)}')

        # Flatten the embeddings structure to match the expected return format
        # Convert from Dict[doc_id, Dict[field_name, List[List[float]]]] 
        # to Dict[doc_id, List[List[float]]]
        doc_vectors: Dict[str, List[List[float]]] = {}
        docs_without_vectors = []
        
        for doc_id, field_embeddings in doc_embeddings_by_field.items():
            vectors: List[List[float]] = []
            
            # Flatten all embeddings from all fields for this document
            for field_name, embedding_list in field_embeddings.items():
                # For legacy unstructured indices, field_name will be "marqo__embeddings"
                # and we should include all embeddings regardless of tensor_fields filter
                # since all embeddings are stored together in marqo__embeddings
                if (tensor_fields is None or 
                    field_name in tensor_fields or
                    (marqo_index.type == IndexType.Unstructured and
                     field_name == unstructured_common.VESPA_DOC_EMBEDDINGS)):
                    vectors.extend(embedding_list)
            
            doc_vectors[doc_id] = vectors

            if len(vectors) == 0:
                docs_without_vectors.append(doc_id)

        if len(docs_without_vectors) > 0:
            raise InvalidArgumentError(
                f'The following documents do not have embeddings: {", ".join(docs_without_vectors)}'
            )

        return doc_vectors


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
                  score_modifiers: Optional[ScoreModifierLists] = None,
                  rerank_depth: Optional[int] = None
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
            rerank_depth: Rerank depth
        """
        # TODO - The dependence on Config in tensor_search is bad design. Refactor to require specific dependencies
        from marqo import config
        from marqo.tensor_search import tensor_search, index_meta_cache

        t0 = timer()

        marqo_index = index_meta_cache.get_index(index_management=self.index_management, index_name=index_name)

        if interpolation_method is None:
            interpolation_method = self.get_default_interpolation_method(marqo_index, documents)

        vector_interpolation = from_interpolation_method(interpolation_method)

        # Get document vectors using the helper method
        doc_vectors = self.get_doc_vectors_from_ids(
            index_name=index_name,
            documents=documents,
            tensor_fields=tensor_fields
        )

        # Save original document IDs for filtering
        if isinstance(documents, dict):
            all_document_ids = list(documents.keys())
        else:
            all_document_ids = documents

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
        except AllZeroWeightsError as e:
            raise InvalidArgumentError(
                f'Cannot interpolate vectors with all zero weights. '
                'Please ensure at least one weight is non-zero.'
            )
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
            recommend_filter = self.get_exclusion_filter(marqo_index, all_document_ids, filter)
        else:
            recommend_filter = filter

        results = tensor_search.search(
            config.Config(self.vespa_client, inference=self.inference),
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
            processing_start=t0,
            rerank_depth=rerank_depth
        )

        return results

    def get_default_interpolation_method(self, marqo_index: MarqoIndex,
                                         context_documents: Union[List[str], Dict[str, float]]) -> InterpolationMethod:
        """
        Returns the default interpolation method based on the index configuration and whether context documents
        exist. For recommend endpoint, context documents always exist.

        For indexes that normalize embeddings, SLERP is used if context documents are provided, NLERP is used
        otherwise (None).
        """
        if marqo_index.normalize_embeddings:
            if context_documents is not None:
                return InterpolationMethod.SLERP
            else:
                # NLERP is used to preserve existing search behavior with no context docs.
                return InterpolationMethod.NLERP
        else:
            return InterpolationMethod.LERP

    def get_exclusion_filter(self, marqo_index: MarqoIndex, documents: List[str], user_filter: Optional[str]) -> str:
        if marqo_index.type == IndexType.Structured:
            not_in = 'NOT _id IN (' + ', '.join([f'{doc}' for doc in documents]) + ')'
        else:
            not_in = 'NOT (' + ' OR '.join([f'_id:({doc})' for doc in documents]) + ')'

        if user_filter is not None and user_filter.strip() != '':
            return f'({user_filter}) AND {not_in}'
        else:
            return not_in
