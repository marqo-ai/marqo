from typing import Dict, List, Union, Optional

from marqo import config
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.utils.vector_interpolation import Slerp
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
                  exclude_input_documents=True,
                  interpolation_method: str = "nlerp"
                  ):
        from marqo.tensor_search import tensor_search

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
            if isinstance(documents , dict):
                weight = documents[document_id]
            else:
                weight = 1
            vectors.extend(vector_list)
            weights.extend([weight] * len(vector_list))

        vector_interpolation = Slerp()

        interpolated_vector = vector_interpolation.interpolate(
            vectors, weights
        )

        results = tensor_search.search(
            config.Config(self.vespa_client),
            index_name,
            query=interpolated_vector,
            searchable_attributes=searchable_attributes
        )


        pass
