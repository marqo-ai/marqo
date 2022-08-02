import functools
import json
import logging
from urllib import parse
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Union
from marqo.neural_search import neural_search
from marqo._httprequests import HttpRequests
from marqo.config import Config
from marqo.neural_search.enums import MediaType
from marqo.marqo_logging import logger
from marqo.enums import SearchMethods, Devices
from marqo import errors

# pylint: disable=too-many-public-methods
class Index():
    """
    Indexes routes wrapper.

    Index class gives access to all indexes routes and child routes (inherited).
    https://docs.marqo.com/reference/api/indexes.html
    """

    def __init__(
        self,
        config: Config,
        index_name: str,
        primary_key: Optional[str] = None,
        created_at: Optional[Union[datetime, str]] = None,
        updated_at: Optional[Union[datetime, str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        config:
            Config object containing permission and location of marqo.
        index_name:
            UID of the index on which to perform the index actions.
        primary_key:
            Primary-key of the index.
        """
        self.config = config
        self.http = HttpRequests(config)
        self.index_name = index_name
        self.primary_key = primary_key
        self.created_at = self._maybe_datetime(created_at)
        self.updated_at = self._maybe_datetime(updated_at)

    def delete(self) -> Dict[str, Any]:
        """Delete the index.

        Raises
        ------
        s2SearchApiError
            An error containing details about why marqo can't process your request. marqo error codes are described here: https://docs.marqo.com/errors/#marqo-errors
        """
        return neural_search.delete_index(config=self.config, index_name=self.index_name)

    @staticmethod
    def create(config: Config, index_name: str,
               treat_urls_and_pointers_as_images=False,
               model=None,
               normalize_embeddings=True,
               sentences_per_chunk=2,
               sentence_overlap=0
               ) -> Dict[str, Any]:
        """Create the index.

        Parameters
        ----------
        config:
            config instance
        index_name:
            name of the index.

        Returns
        -------
        task:
            Dictionary containing a task to track the informations about the progress of an asynchronous process.
            https://docs.marqo.com/reference/api/tasks.html#get-one-task

        Raises
        ------
        s2SearchApiError
            An error containing details about why marqo can't process your request. marqo error codes are described here: https://docs.marqo.com/errors/#marqo-errors
        """
        return neural_search.create_vector_index(
            config=config, index_name=index_name, media_type=MediaType.default, neural_settings={
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": treat_urls_and_pointers_as_images,
                    "model": model,
                    "normalize_embeddings": normalize_embeddings,
                    "text_preprocessing": {
                        "split_overlap": sentence_overlap,
                        "split_length": sentences_per_chunk,
                        "split_method": "sentence"
                    }
                }
            }
        )

    def refresh(self):
        """refreshes the index"""
        return self.http.post(path=F"{self.index_name}/_refresh")

    def search(self, q: str, searchable_attributes: Optional[List[str]]=None,
               limit: int=10, search_method: Union[SearchMethods.NEURAL, str] = SearchMethods.NEURAL,
               highlights=True
               ) -> Dict[str, Any]:
        """Search in the index.

        Parameters
        ----------
        query:
            String containing the searched word(s)
        opt_params (optional):
            Dictionary containing optional query parameters
            https://docs.marqo.com/reference/api/search.html#search-in-an-index
        searchable_attributes: a subset of attributes to search through

        Returns
        -------
        results:
            Dictionary with hits, offset, limit, processingTime and initial query

        Raises
        ------
        s2SearchApiError
            An error containing details about why marqo can't process your request. marqo error codes are described here: https://docs.marqo.com/errors/#marqo-errors
        """
        return neural_search.search(
            config=self.config, index_name=self.index_name, text=q, return_doc_ids=True,
            searchable_attributes=searchable_attributes, search_method=search_method, result_count=limit,
            highlights=highlights
        )

    def get_document(self, document_id: Union[str, int]) -> Dict[str, Any]:
        """Get one document with given document identifier.

        Parameters
        ----------
        document_id:
            Unique identifier of the document.

        Returns
        -------
        document:
            Dictionary containing the documents information.

        Raises
        ------
        s2SearchApiError
            An error containing details about why marqo can't process your request. marqo error codes are described here: https://docs.marqo.com/errors/#marqo-errors
        """
        return neural_search.get_document_by_id(
            config=self.config, index_name=self.index_name, document_id=document_id)

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        auto_refresh=True,
        batch_size: int = None,
        use_parallel: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Add documents to the index.

        Parameters
        ----------
        documents:
            List of documents. Each document should be a dictionary.

        auto_refresh:
            Automatically refresh the index. If you are making lots of requests, it is advised to turn this to
            false to increase performance.

        apply_batching: if True, the documents will be batched into sizes of 'batch_size'

        batch_size: if batch_documents is True, documents will be indexed into batches of this size

        Returns
        -------
        We need to translate into this endpoint: https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-bulk.html

        See Jesse's code for JSONL: https://github.com/S2Search/NeuralSearchPOC/blob/main/data/csv_to_json.py#L152
        Raises
        ------
        s2SearchApiError
            An error containing details about why marqo can't process your request. marqo error codes are described here: https://docs.marqo.com/errors/#marqo-errors
        """

        if batch_size is None:
            return neural_search.add_documents(
                config=self.config, index_name=self.index_name, docs=documents, auto_refresh=auto_refresh)
        elif use_parallel:
            from marqo import parallel
            return parallel.add_documents_mp(config=self.config, index_name=self.index_name, docs=documents, auto_refresh=auto_refresh, batch_size=batch_size)
        else:
            if batch_size <= 0:
                raise errors.MarqoError("Batch size can't be less than 1!")
            return self._batch_request(dataset=documents,batch_size=batch_size, verbose=False)

    def delete_documents(self, ids: List[str], auto_refresh=True) -> Dict[str, int]:
        """Delete multiple documents from the index.

        Parameters
        ----------
        ids:
            List of unique identifiers of documents.
        auto_refresh:
            if true refreshes the index
        Returns
        -------
        task:
            Dictionary containing a task to track the informations about the progress of an asynchronous process.
            https://docs.marqo.com/reference/api/tasks.html#get-one-task

        Raises
        ------
        s2SearchApiError
            An error containing details about why marqo can't process your request. marqo error codes are described here: https://docs.marqo.com/errors/#marqo-errors
        """
        return neural_search.delete_documents(
            config=self.config, index_name=self.index_name, doc_ids=ids,
            auto_refresh=auto_refresh)

    def get_stats(self) -> Dict[str, Any]:
        """Get stats of the index"""
        return neural_search.get_stats(config=self.config, index_name=self.index_name)

    @staticmethod
    def _maybe_datetime(the_date: Optional[Union[datetime, str]]) -> Optional[datetime]:
        """This should handle incoming timestamps from S2Search, including
         parsing if necessary."""
        if the_date is None or not the_date:
            return None

        if isinstance(the_date, datetime):
            return the_date
        elif isinstance(the_date, str):
            parsed_date = datetime.strptime(the_date, "%Y-%m-%dT%H:%M:%S.%f")
            return parsed_date

    def _batch_request(self, dataset, batch_size=100, verbose=True) -> List[Dict[str, Any]]:
        """Batch by the number of documents"""
        logger.info(f"starting batch ingestion in sizes of {batch_size}")

        deeper = ((doc, i, batch_size) for i, doc in enumerate(dataset))

        def batch_requests(gathered, doc_tuple):
            doc, i, the_batch_size = doc_tuple
            if i % the_batch_size == 0:
                gathered.append([doc,])
            else:
                gathered[-1].append(doc)
            return gathered

        batched = functools.reduce(lambda x, y: batch_requests(x, y), deeper, [])

        def verbosely_add_docs(i, docs):
            t0 = datetime.now()
            res = neural_search.add_documents(
                config=self.config, index_name=self.index_name,
                docs=docs, auto_refresh=False)
            total_batch_time = datetime.now() - t0
            num_docs = len(docs)

            logger.info(f"    batch {i}: ingested {num_docs} docs. Time taken: {total_batch_time}. "
                        f"Average timer per doc {total_batch_time/num_docs}")
            if verbose:
                logger.info(f"        results from indexing batch {i}: {res}")
            return res

        results = [verbosely_add_docs(i, docs) for i, docs in enumerate(batched)]
        logger.info('completed batch ingestion.')
        return results

######################################## Class functionality not included in this version#################

    # def add_documents_json(
    #     self,
    #     str_documents: str,
    #     primary_key: Optional[str] = None,
    # ) -> Dict[str, Any]:

    # def add_documents_raw(
    #     self,
    #     str_documents: str,
    #     primary_key: Optional[str] = None,
    #     content_type: Optional[str] = None,
    # ) -> Dict[str, Any]:

    # def add_documents_from_csv_file(
    #         self,
    #         str_documents: str,
    #         primary_key: Optional[str] = None,
    # ) -> Dict[str, Any]:
    #     pass

    # def update_documents(
    #     self,
    #     documents: List[Dict[str, Any]],
    #     primary_key: Optional[str] = None
    # ) -> Dict[str, Any]:
    #     """Update documents in the index.
    #     This can be done using regular add_documents
    #     """

    # def update(self, primary_key: str) -> Dict[str, Any]:
    #     """Update the index primary-key."""

    # def fetch_info(self) -> 'Index':
    #     """Fetch the info of the index. """

    # def get_primary_key(self) -> Optional[str]:
    #     """Get the primary key.
    #     Meilisearch specific. In Meilisearch, users decide which field acts as the ID
    #     """

    # def get_tasks(self) -> Dict[str, List[Dict[str, Any]]]:
    #     """Get all tasks of a specific index from the last one"""

    # def get_task(self, uid: int) -> Dict[str, Any]:
    #     """Get one task through the route of a specific index. """

    # def wait_for_task(
    #     self, uid: int,
    #     timeout_in_ms: int = 5000,
    #     interval_in_ms: int = 50,
    # ) -> Dict[str, Any]:
    #     """Wait until marqo processes a task until it fails or succeeds. """



    # def get_documents(self, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    #     """Get a set of documents from the index.
    #     Will be implemented in near future, probably...
    #       https://docs.meilisearch.com/reference/api/documents.html#get-documents
    #     """

    # def add_documents_in_batches(
    #     self,
    #     documents: List[Dict[str, Any]],
    #     batch_size: int = 1000,
    #     primary_key: Optional[str] = None,
    # ) -> List[Dict[str, Any]]:
    #     """Add documents to the index in batches.
    #     Will probably be implemented later. Seems useful.
    #     """

    # def add_documents_csv(
    #     self,
    #     str_documents: str,
    #     primary_key: Optional[str] = None,
    # ) -> Dict[str, Any]:
    #     """Add string documents from a CSV file to the index."""

    # def add_documents_ndjson(
    #     self,
    #     str_documents: str,
    #     primary_key: Optional[str] = None,
    # ) -> Dict[str, Any]:
    #     """Add string documents from a NDJSON file to the index."""

    # def update_documents_in_batches(
    #     self,
    #     documents: List[Dict[str, Any]],
    #     batch_size: int = 1000,
    #     primary_key: Optional[str] = None
    # ) -> List[Dict[str, Any]]:
    #     """Update documents to the index in batches."""

    # def delete_all_documents(self) -> Dict[str, int]:
    #     """Delete all documents from the index. """

    # # GENERAL SETTINGS ROUTES

    # def get_settings(self) -> Dict[str, Any]:
    #     """Get settings of the index."""
    #
    # def update_settings(self, body: Dict[str, Any]) -> Dict[str, Any]:
    #     """Update settings of the index. """
    #
    # def reset_settings(self) -> Dict[str, Any]:
    #     """Reset settings of the index to default values."""

    # # RANKING RULES SUB-ROUTES

    # def get_ranking_rules(self) -> List[str]:
    #     """Get ranking rules of the index."""
    #
    # def update_ranking_rules(self, body: List[str]) -> Dict[str, Any]:
    #     """Update ranking rules of the index."""
    #
    # def reset_ranking_rules(self) -> Dict[str, Any]:
    #     """Reset ranking rules of the index to default values."""

    # # DISTINCT ATTRIBUTE SUB-ROUTES

    # def get_distinct_attribute(self) -> Optional[str]:
    #     """Get distinct attribute of the index. """
    #
    # def update_distinct_attribute(self, body: Dict[str, Any]) -> Dict[str, Any]:
    #     """Update distinct attribute of the index."""
    #
    # def reset_distinct_attribute(self) -> Dict[str, Any]:
    #     """Reset distinct attribute of the index to default values."""

    # # SEARCHABLE ATTRIBUTES SUB-ROUTES

    # def get_searchable_attributes(self) -> List[str]:
    #     """Get searchable attributes of the index."""
    #
    # def update_searchable_attributes(self, body: List[str]) -> Dict[str, Any]:
    #     """Update searchable attributes of the index."""
    #
    # def reset_searchable_attributes(self) -> Dict[str, Any]:
    #     """Reset searchable attributes of the index to default values."""

    # # DISPLAYED ATTRIBUTES SUB-ROUTES

    # def get_displayed_attributes(self) -> List[str]:
    #     """Get displayed attributes of the index."""
    #
    # def update_displayed_attributes(self, body: List[str]) -> Dict[str, Any]:
    #     """Update displayed attributes of the index. """
    #
    # def reset_displayed_attributes(self) -> Dict[str, Any]:
    #     """Reset displayed attributes of the index to default values. """

    # # STOP WORDS SUB-ROUTES

    # def get_stop_words(self) -> List[str]:
    #     """Get stop words of the index."""
    #
    # def update_stop_words(self, body: List[str]) -> Dict[str, Any]:
    #     """Update stop words of the index."""
    #
    # def reset_stop_words(self) -> Dict[str, Any]:
    #     """Reset stop words of the index to default values."""

    # # SYNONYMS SUB-ROUTES

    # def get_synonyms(self) -> Dict[str, List[str]]:
    #     """Get synonyms of the index."""
    #
    # def update_synonyms(self, body: Dict[str, List[str]]) -> Dict[str, Any]:
    #     """Update synonyms of the index."""
    #
    # def reset_synonyms(self) -> Dict[str, Any]:
    #     """Reset synonyms of the index to default values."""

    # # FILTERABLE ATTRIBUTES SUB-ROUTES

    # def get_filterable_attributes(self) -> List[str]:
    #     """Get filterable attributes of the index."""
    #
    # def update_filterable_attributes(self, body: List[str]) -> Dict[str, Any]:
    #     """Update filterable attributes of the index. """
    #
    # def reset_filterable_attributes(self) -> Dict[str, Any]:
    #     """Reset filterable attributes of the index to default values."""

    # # SORTABLE ATTRIBUTES SUB-ROUTES

    # def get_sortable_attributes(self) -> List[str]:
    #     """Get sortable attributes of the index."""
    #
    # def update_sortable_attributes(self, body: List[str]) -> Dict[str, Any]:
    #     """Update sortable attributes of the index."""
    #
    # def reset_sortable_attributes(self) -> Dict[str, Any]:
    #     """Reset sortable attributes of the index to default values."""

    # # TYPO TOLERANCE SUB-ROUTES

    # def get_typo_tolerance(self) -> Dict[str, Any]:
    #     """Get typo tolerance of the index."""
    #
    # def update_typo_tolerance(self, body: Dict[str, Any]) -> Dict[str, Any]:
    #     """Update typo tolerance of the index"""
    #
    # def reset_typo_tolerance(self) -> Dict[str, Any]:
    #     """Reset typo tolerance of the index to default values."""

    # # HELPERS

    # @staticmethod
    # def _batch(
    #         documents: List[Dict[str, Any]], batch_size: int
    # ) -> Generator[List[Dict[str, Any]], None, None]:
    #     total_len = len(documents)
    #     for i in range(0, total_len, batch_size):
    #         yield documents[i: i + batch_size]

    # def __settings_url_for(self, sub_route: str) -> str:
    #     return f'{self.config.paths.index}/{self.index_name}/{self.config.paths.setting}/{sub_route}'
