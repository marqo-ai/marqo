import uuid

from marqo.client import Client
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase, TestImageUrls
import base64
import requests


class TestSearchCommon(MarqoTestCase):
    """A class to test common search functionalities for structured and unstructured indexes.

    We should test the shared functionalities between structured and unstructured indexes here to avoid code duplication
    and branching in the test cases."""

    structured_text_index_name = "structured_index_text" + str(uuid.uuid4()).replace('-', '')
    structured_image_index_name = "structured_image_index" + str(uuid.uuid4()).replace('-', '')
    structured_filter_index_name = "structured_filter_index" + str(uuid.uuid4()).replace('-', '')

    unstructured_text_index_name = "unstructured_index_text" + str(uuid.uuid4()).replace('-', '')
    unstructured_image_index_name = "unstructured_image_index" + str(uuid.uuid4()).replace('-', '')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = Client(**cls.client_settings)

        cls.create_indexes([
            {
                "indexName": cls.structured_text_index_name,
                "type": "structured",
                "model": "hf/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "title", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "content", "type": "text", "features": ["filter", "lexical_search"]},
                ],
                "tensorFields": ["title", "content"],
            },
            {
                "indexName": cls.structured_filter_index_name,
                "type": "structured",
                "model": "hf/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "field_a", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "field_b", "type": "text", "features": ["filter"]},
                    {"name": "str_for_filtering", "type": "text", "features": ["filter"]},
                    {"name": "int_for_filtering", "type": "int", "features": ["filter"]},
                    {"name": "long_field_1", "type": "long", "features": ["filter"]},
                    {"name": "double_field_1", "type": "double", "features": ["filter"]},
                    {"name": "array_long_field_1", "type": "array<long>", "features": ["filter"]},
                    {"name": "array_double_field_1", "type": "array<double>", "features": ["filter"]}
                ],
                "tensorFields": ["field_a", "field_b"],
            },
            {
                "indexName": cls.structured_image_index_name,
                "type": "structured",
                "model": "open_clip/ViT-B-32/laion2b_s34b_b79k",
                "allFields": [
                    {"name": "title", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "content", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "text_field_1", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "image_content", "type": "image_pointer"},
                    {"name": "image_field_1", "type": "image_pointer"},
                ],
                "tensorFields": ["title", "image_content", "image_field_1"],
            }
        ])

        cls.create_indexes([
            {
                "indexName": cls.unstructured_text_index_name,
                "type": "unstructured",
                "model": "hf/all-MiniLM-L6-v2",
            },
            {
                "indexName": cls.unstructured_image_index_name,
                "type": "unstructured",
                "model": "open_clip/ViT-B-32/laion2b_s34b_b79k"
            }
        ])

        cls.indexes_to_delete = [cls.structured_image_index_name, cls.structured_filter_index_name,
                                 cls.structured_text_index_name, cls.unstructured_image_index_name,
                                 cls.unstructured_text_index_name]

    def test_lexical_query_can_not_be_none(self):
        context = {"tensor": [{"vector": [1, ] * 384, "weight": 1},
                              {"vector": [2, ] * 384, "weight": 2}]}

        test_case = [
            (None, context, "with context"),
            (None, None, "without context")
        ]
        for index_name in [self.structured_text_index_name, self.unstructured_image_index_name]:
            for query, context, msg in test_case:
                with self.subTest(f"{index_name} - {msg}"):
                    with self.assertRaises(MarqoWebError) as e:
                        res = self.client.index(index_name).search(q=None, context=context, search_method="LEXICAL")
                    self.assertIn("Query(q) is required for lexical search", str(e.exception.message))

    def test_tensor_search_query_can_be_none(self):
        context = {"tensor": [{"vector": [1, ] * 384, "weight": 1},
                              {"vector": [2, ] * 384, "weight": 2}]}
        for index_name in [self.structured_text_index_name, self.unstructured_text_index_name]:
            res = self.client.index(index_name).search(q=None, context=context)
            self.assertIn("hits", res)

    def test_add_document_and_search_for_private_images(self):
        documents = [
            {
                "image_field_1": "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small.png",
                "text_field_1": "A private image with a png extension",
                "_id": "1"
            },
            {
                "image_field_1": "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small",
                "text_field_1": "A private image without an extension",
                "_id": "2"
            }
        ]

        kwargs_list = [
            {"media_download_headers": {"marqo_media_header": "media_header_test_key"}},
            {"image_download_headers": {"marqo_media_header": "media_header_test_key"}}
        ]

        for index_name in [self.unstructured_image_index_name, self.structured_image_index_name]:
            tensor_fields = ["image_field_1"] if (
                    index_name == self.unstructured_image_index_name) else None
            res = self.client.index(index_name).add_documents(
                documents, tensor_fields=tensor_fields,
                media_download_headers={"marqo_media_header": "media_header_test_key"}
            )

            for kwargs in kwargs_list:
                for query in [
                    "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small",
                    {
                        "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small.png": 1,
                        "A private image without an extension": 1
                    }
                ]:
                    with self.subTest(f"{index_name} - {kwargs} - {query}"):
                        res = self.client.index(index_name).search(query, **kwargs)
                        self.assertIn("hits", res, res)
                        self.assertEqual(2, len(res["hits"]), res)

    def test_invalidArgError_is_raised_when_searching_a_private_image(self):
        query = "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small"
        for index_name in [self.structured_image_index_name, self.unstructured_image_index_name]:
            with self.subTest(f"{index_name}"):
                with self.assertRaises(MarqoWebError) as e:
                    self.client.index(index_name).search(query)
                self.assertIn("Error downloading media file", str(e.exception))

    def test_proper_error_if_both_imageDownloadHeaders_and_mediaDownloadHeaders_are_provided(self):
        """Test that an error is raised if both imageDownloadHeaders and mediaDownloadHeaders are provided."""
        for index_name in [self.unstructured_image_index_name, self.structured_image_index_name]:
            with self.assertRaises(MarqoWebError) as cm:
                res = self.client.index(index_name).search(
                    "test",
                    image_download_headers={"marqo_media_header": "media_header_test_key"},
                    media_download_headers={"marqo_media_header": "media_header_test_key"}
                )
                self.assertIn("Cannot set both imageDownloadHeaders and mediaDownloadHeaders.",
                              str(cm.exception.message))

    def test_rerank_depth(self):
        """Test rerank_depth behavior in TENSOR search."""
        for index_name in [self.unstructured_image_index_name, self.structured_image_index_name]:
            with self.subTest(index=index_name):
                docs = [{
                    "title": f"Doc {i}",
                    "content": "some extra info",
                    "_id": str(i)
                } for i in range(10)]
                tensor_fields = ["title", "content"] if index_name == self.unstructured_image_index_name else None

                add_res = self.client.index(index_name).add_documents(docs, tensor_fields=tensor_fields)
                if add_res["errors"]:
                    raise Exception(f"Failed to add docs to index {index_name}")

                # Case 1: rerank_depth < result_count → limit overrides rerank_depth
                with self.subTest(case="rerank_depth_smaller_than_limit"):
                    res = self.client.index(index_name).search(
                        q="Doc", rerank_depth=5, limit=10, search_method="TENSOR"
                    )
                    self.assertEqual(len(res["hits"]), 10)

                # Case 2: rerank_depth is negative → error expected
                with self.subTest(case="invalid_negative_rerank_depth"):
                    with self.assertRaises(MarqoWebError):
                        self.client.index(index_name).search(
                            q="Doc", rerank_depth=-1, limit=10, search_method="TENSOR"
                        )

                # Case 3: no rerank_depth → should return full limit
                with self.subTest(case="no_rerank_depth"):
                    res = self.client.index(index_name).search(
                        q="Doc", limit=10, search_method="TENSOR"
                    )
                    self.assertEqual(len(res["hits"]), 10)

    def test_rerank_depth_hybrid(self):
        """Test rerankDepthTensor behavior in HYBRID search."""
        for index_name in [self.unstructured_image_index_name, self.structured_image_index_name]:
            with self.subTest(index=index_name):
                docs = [{
                    "title": f"Doc {i}",
                    "content": "some extra info",
                    "_id": str(i)
                } for i in range(10)]
                tensor_fields = ["title", "content"] if index_name == self.unstructured_image_index_name else None

                add_res = self.client.index(index_name).add_documents(docs, tensor_fields=tensor_fields)
                if add_res["errors"]:
                    raise Exception(f"Failed to add docs to index {index_name}")

                # Case 1: rerankDepthTensor < result_count → rerank_depth is overridden
                with self.subTest(case="rerank_depth_tensor_less_than_limit"):
                    res = self.client.index(index_name).search(
                        q="Doc", limit=10, search_method="HYBRID", hybrid_parameters={
                            "retrievalMethod": "tensor",
                            "rankingMethod": "tensor",
                            "rerankDepthTensor": 5
                        }
                    )
                    self.assertEqual(len(res["hits"]), 10)

                # Case 2: rerankDepthTensor is negative → raises error
                with self.subTest(case="invalid_negative_rerank_depth_tensor"):
                    with self.assertRaises(MarqoWebError):
                        self.client.index(index_name).search(
                            q="Doc", limit=10, search_method="HYBRID", hybrid_parameters={
                                "retrievalMethod": "tensor",
                                "rankingMethod": "tensor",
                                "rerankDepthTensor": -1
                            }
                        )

                # Case 3: No rerankDepthTensor → should return full limit
                with self.subTest(case="no_rerank_depth_tensor"):
                    res = self.client.index(index_name).search(
                        q="Doc", limit=10, search_method="HYBRID", hybrid_parameters={
                            "retrievalMethod": "tensor",
                            "rankingMethod": "tensor"
                        }
                    )
                    self.assertEqual(len(res["hits"]), 10)

    def test_hybrid_search_validations(self):
        # Add docs
        docs = [
            {
                "title": "Cool Document 1",
                "content": "some extra info",
                "_id": "1"
            },
            {
                "title": "Just Your Average Doc",
                "content": "this is a solid doc",
                "_id": "2"
            }
        ]
        for index_name in [self.structured_text_index_name, self.unstructured_text_index_name]:
            with self.subTest(index_name):
                self.client.index(index_name).add_documents(
                    docs,
                    tensor_fields=["title", "content"] if index_name == self.unstructured_text_index_name else None
                )
                # Hybrid search with no query or context should raise an error
                with self.subTest("Hybrid search with no query or context"):
                    with self.assertRaises(MarqoWebError) as e:
                        self.client.index(index_name).search(
                            search_method="HYBRID"
                        )
                    assert e.exception.status_code == 422
                    assert "One of Query(q), context, hybridParameters.queryTensor, or hybridParameters.queryTensor is required for HYBRID search but all are missing" in str(
                        e.exception)

                with self.subTest("Hybrid search with no query or context should raise an error"):
                    with self.assertRaises(MarqoWebError) as e:
                        self.client.index(index_name).search(
                            search_method="HYBRID",
                            hybrid_parameters={}
                        )
                    assert e.exception.status_code == 422
                    assert "One of Query(q), context, hybridParameters.queryTensor, or hybridParameters.queryTensor is required for HYBRID search but all are missing" in str(
                        e.exception)

                with self.subTest("Hybrid search with query and queryTensor/queryLexical should raise an error"):
                    with self.assertRaises(MarqoWebError) as e:
                        self.client.index(index_name).search(
                            q="Cool",
                            search_method="HYBRID",
                            hybrid_parameters={
                                "queryTensor": {"Cool": 1},
                            }
                        )
                    assert e.exception.status_code == 422
                    assert "Query(q) cannot be provided for HYBRID search when hybridParameters.queryTensor or hybridParameters.queryLexical is provided" in str(
                        e.exception)
                    with self.assertRaises(MarqoWebError) as e:
                        self.client.index(index_name).search(
                            q="Cool",
                            search_method="HYBRID",
                            hybrid_parameters={
                                "queryLexical": "Cool",
                            }
                        )
                    assert e.exception.status_code == 422
                    assert "Query(q) cannot be provided for HYBRID search when hybridParameters.queryTensor or hybridParameters.queryLexical is provided" in str(
                        e.exception)

                with self.subTest(
                        "Hybrid search with only one queryTensor/queryLexical and retrievalMethod=disjunction raises an error"):
                    with self.assertRaises(MarqoWebError) as e:
                        self.client.index(index_name).search(
                            search_method="HYBRID",
                            hybrid_parameters={
                                "queryTensor": {"Cool": 1},
                            }
                        )
                    assert e.exception.status_code == 400
                    assert "Either 'hybridParameters.queryLexical' or just 'q'" in str(e.exception)
                    with self.assertRaises(MarqoWebError) as e:
                        self.client.index(index_name).search(
                            search_method="HYBRID",
                            hybrid_parameters={
                                "queryLexical": "Cool",
                            }
                        )
                    assert e.exception.status_code == 400
                    assert "Marqo could not collect any vectors from the search query" in str(e.exception)

                with self.subTest(
                        "Hybrid search without query and with queryTensor/queryLexical should not raise an error"):
                    self.client.index(index_name).search(
                        search_method="HYBRID",
                        hybrid_parameters={
                            "queryTensor": {"Cool": 1},
                            "retrievalMethod": "tensor",
                            "rankingMethod": "tensor"
                        }
                    )
                    self.client.index(index_name).search(
                        search_method="HYBRID",
                        hybrid_parameters={
                            "queryLexical": "Cool",
                            "retrievalMethod": "lexical",
                            "rankingMethod": "lexical"
                        }
                    )
                    self.client.index(index_name).search(
                        search_method="HYBRID",
                        hybrid_parameters={
                            "queryTensor": {"Cool": 1},
                            "queryLexical": "Cool",
                        }
                    )

                with self.subTest(
                        "Hybrid search with lexicalQuery and retrieval/ranking methods 'Tensor' should raise an error"):
                    with self.assertRaises(MarqoWebError) as e:
                        self.client.index(index_name).search(
                            search_method="HYBRID",
                            hybrid_parameters={
                                "queryLexical": "Cool",
                                "retrievalMethod": "tensor",
                                "rankingMethod": "tensor"
                            }
                        )
                    assert e.exception.status_code == 400
                    assert "'hybridParameters.queryLexical' cannot be provided when 'retrievalMethod' and 'rankingMethod' are both 'tensor'." in str(
                        e.exception)

                with self.subTest(
                        "Hybrid search with tensorQuery and retrieval/ranking methods 'Lexical' should raise an error"):
                    with self.assertRaises(MarqoWebError) as e:
                        self.client.index(index_name).search(
                            search_method="HYBRID",
                            hybrid_parameters={
                                "queryTensor": {"Cool": 1},
                                "retrievalMethod": "lexical",
                                "rankingMethod": "lexical"
                            }
                        )
                    assert e.exception.status_code == 400
                    assert "'hybridParameters.queryTensor' cannot be provided when 'retrievalMethod' and 'rankingMethod' are both 'lexical'." in str(e.exception)

    def test_search_with_context_documents(self):
        for index_name in [self.unstructured_text_index_name, self.structured_text_index_name]:
            with self.subTest(index=index_name):
                self.client.index(index_name=index_name).add_documents(
                    [
                        {
                            "title": "A comparison of the best pets",
                            "content": "Animals",
                            "_id": "d1"
                        },
                        {
                            "title": "The history of dogs",
                            "content": "A history of household pets",
                            "_id": "d2"
                        }
                    ],
                    tensor_fields=["title", "content"] if index_name == self.unstructured_text_index_name else None
                )

                for interpolation_method in ["lerp", "nlerp", "slerp"]:
                    context = {
                        "documents": {
                            "parameters": {
                                "excludeInputDocuments": False,
                                "tensorFields": ["title", "content"]
                            },
                            "ids": {
                                "d1": 1
                            }
                        }
                    }

                    res = self.client.index(index_name).search(q={"best pets": 1},
                                                                context=context,
                                                                search_method="TENSOR",
                                                                interpolation_method=interpolation_method
                                                                )
                    self.assertEqual(res["hits"][0]["_id"], "d1")

    def test_hybrid_search_with_context_documents(self):
        """
        Tests all valid combinations of retrieval and ranking methods for hybrid search with context documents.
        Shows the search runs without errors and correctly retrieves the expected document.
        """
        for index_name in [self.unstructured_text_index_name, self.structured_text_index_name]:
            with self.subTest(index=index_name):
                self.client.index(index_name=index_name).add_documents(
                    [
                        {
                            "title": "A comparison of the best pets",
                            "content": "Animals",
                            "_id": "d1"
                        },
                        {
                            "title": "The history of dogs",
                            "content": "A history of household pets",
                            "_id": "d2"
                        }
                    ],
                    tensor_fields=["title", "content"] if index_name == self.unstructured_text_index_name else None
                )

                test_cases = [
                    ("disjunction", "rrf", {"best pets": 1}, "animals"),
                    ("tensor", "tensor", {"best pets": 1}, None),
                    ("tensor", "lexical", {"best pets": 1}, "animals"),
                    ("lexical", "tensor", {"best pets": 1}, "animals")
                ]

                for interpolation_method in ["lerp", "nlerp", "slerp"]:
                    context = {
                        "documents": {
                            "parameters": {
                                "excludeInputDocuments": False,
                                "tensorFields": ["title", "content"]
                            },
                            "ids": {
                                "d1": 1
                            }
                        }
                    }

                    for retrieval_method, ranking_method, query_tensor, query_lexical in test_cases:
                        with self.subTest(retrieval_method=retrieval_method, ranking_method=ranking_method):
                            res = self.client.index(index_name).search(
                                q=None,
                                context=context,
                                search_method="HYBRID",
                                hybrid_parameters={
                                    "queryTensor": query_tensor,
                                    "queryLexical": query_lexical,
                                    "retrievalMethod": retrieval_method,
                                    "rankingMethod": ranking_method
                                },
                                interpolation_method=interpolation_method
                            )
                            self.assertEqual(res["hits"][0]["_id"], "d1")

    def test_approximate_threshold_success(self):
        """Test approximate threshold parameter success cases with result comparison."""
        # Add 100 documents - 50 with content:small, 50 with content:large
        docs = []
        # First 50 docs with content:small
        for i in range(50):
            docs.append({
                '_id': str(i + 1),
                'title': 'This is a test document.',
                'content': 'small'
            })
        # Next 50 docs with content:large
        for i in range(50, 100):
            docs.append({
                '_id': str(i + 1),
                'title': 'This is a test document.',
                'content': 'large'
            })

        for index_name in [self.structured_text_index_name,
                           self.unstructured_text_index_name]:
            with self.subTest(index_name=index_name):
                # Add documents to index
                tensor_fields = (['title'] if
                                 index_name == self.unstructured_text_index_name
                                 else None)
                self.client.index(index_name).add_documents(
                    docs, tensor_fields=tensor_fields
                )

                # Test for TENSOR and HYBRID search methods
                search_methods = ["TENSOR", "HYBRID"]
                for search_method in search_methods:
                    with self.subTest(f"{search_method} search"):
                        # Get baseline results without approximate threshold
                        with self.subTest(f"{search_method} baseline"):
                            baseline_res = self.client.index(index_name).search(
                                q="test",
                                search_method=search_method,
                                filter_string="content:small",
                                limit=10
                            )
                            baseline_ids = {hit['_id'] for hit in baseline_res['hits']}

                        # Test threshold 0.0 (< 0.5) - should match baseline
                        with self.subTest(f"{search_method} threshold 0.0"):
                            res_0 = self.client.index(index_name).search(
                                q="test",
                                search_method=search_method,
                                approximate_threshold=0.0,
                                filter_string="content:small",
                                limit=10
                            )
                            ids_0 = {hit['_id'] for hit in res_0['hits']}
                            self.assertEqual(baseline_ids, ids_0,
                                             f"Threshold 0.0 results should match baseline for {search_method}")

                        # Test threshold 0.5 (>= 0.5) - should differ from baseline
                        with self.subTest(f"{search_method} threshold 0.5"):
                            res_05 = self.client.index(index_name).search(
                                q="test",
                                search_method=search_method,
                                approximate_threshold=0.5,
                                filter_string="content:small",
                                limit=10
                            )
                            ids_05 = {hit['_id'] for hit in res_05['hits']}
                            self.assertNotEqual(baseline_ids, ids_05,
                                                f"Threshold 0.5 results should differ from baseline for {search_method}")

                        # Test threshold 1.0 (>= 0.5) - should differ from baseline
                        with self.subTest(f"{search_method} threshold 1.0"):
                            res_1 = self.client.index(index_name).search(
                                q="test",
                                search_method=search_method,
                                approximate_threshold=1.0,
                                filter_string="content:small",
                                limit=10
                            )
                            ids_1 = {hit['_id'] for hit in res_1['hits']}
                            self.assertNotEqual(baseline_ids, ids_1,
                                                f"Threshold 1.0 results should differ from baseline for {search_method}")

    def test_approximate_threshold_failures(self):
        """Test approximate threshold parameter failure cases."""
        # Add test documents
        docs = [
            {
                "title": "Test Document 1",
                "content": "This is a test document with some content",
                "_id": "1"
            },
            {
                "title": "Test Document 2",
                "content": "Another test document with different content",
                "_id": "2"
            }
        ]

        for index_name in [self.structured_text_index_name,
                           self.unstructured_text_index_name]:
            with self.subTest(index_name=index_name):
                # Add documents to index
                tensor_fields = (["title", "content"] if
                                 index_name == self.unstructured_text_index_name
                                 else None)
                self.client.index(index_name).add_documents(
                    docs, tensor_fields=tensor_fields
                )

                # Test invalid approximate threshold values for TENSOR and HYBRID
                search_methods = ["TENSOR", "HYBRID"]
                for search_method in search_methods:
                    # Test invalid approximate threshold values (outside 0-1)
                    invalid_thresholds = [-0.1, -1.0, 1.1, 2.0]
                    for threshold in invalid_thresholds:
                        with self.subTest(f"Invalid {search_method} threshold {threshold}"):
                            with self.assertRaises(MarqoWebError) as e:
                                self.client.index(index_name).search(
                                    q="test",
                                    search_method=search_method,
                                    approximate_threshold=threshold
                                )
                            self.assertEqual(e.exception.status_code, 422)
                            self.assertIn(
                                "'approximateThreshold' must be between 0 and 1",
                                str(e.exception))

                # Test that approximate threshold fails for LEXICAL search
                with self.subTest("LEXICAL search with threshold should fail"):
                    with self.assertRaises(MarqoWebError) as e:
                        self.client.index(index_name).search(
                            q="test",
                            search_method="LEXICAL",
                            approximate_threshold=0.5
                        )
                    self.assertEqual(e.exception.status_code, 422)
                    error_msg = str(e.exception)
                    self.assertIn("'approximateThreshold'", error_msg)
                    self.assertIn("HYBRID", error_msg)
                    self.assertIn("TENSOR", error_msg)

    def test_query_field_returned_and_base64_omitted_all_search_modes(self):
        """Test that query field is returned for all search modes and base64 content is properly omitted."""

        # Create base64 image data from real image URL
        def url_to_base64(url: str):
            """Convert an image URL to base64 data URL format."""
            response = requests.get(url)
            response.raise_for_status()
            base64_data = base64.b64encode(response.content).decode('utf-8')
            # Determine content type from response headers or default to png
            content_type = response.headers.get('content-type', 'image/png')
            return f"data:{content_type};base64,{base64_data}"
        
        base64_image = url_to_base64(TestImageUrls.HIPPO_STATUE.value)
        
        # Test data setup - add some documents first
        documents = [
            {"title": "test document", "content": "sample text content", "_id": "doc1"},
            {"title": "another document", "content": "more text here", "_id": "doc2"}
        ]
        
        # Define test cases covering all search modes and query types
        test_cases = [
            # (search_method, hybrid_parameters, query_types, description)
            ("TENSOR", None, ["string", "dict"], "tensor_default"),
            ("LEXICAL", None, ["string"], "lexical_default"),  # Lexical only supports string queries
            ("HYBRID", {"retrievalMethod": "disjunction", "rankingMethod": "rrf"}, ["string"], "hybrid_disjunction_rrf"),  # Hybrid only supports string in q parameter
            ("HYBRID", {"retrievalMethod": "lexical", "rankingMethod": "lexical"}, ["string"], "hybrid_lexical_lexical"),
            ("HYBRID", {"retrievalMethod": "lexical", "rankingMethod": "tensor"}, ["string"], "hybrid_lexical_tensor"),
            ("HYBRID", {"retrievalMethod": "tensor", "rankingMethod": "lexical"}, ["string"], "hybrid_tensor_lexical"),
            ("HYBRID", {"retrievalMethod": "tensor", "rankingMethod": "tensor"}, ["string"], "hybrid_tensor_tensor"),
        ]
        
        # Test on both structured and unstructured indexes
        for index_name in [self.structured_text_index_name, self.unstructured_text_index_name]:
            with self.subTest(index=index_name):
                # Add documents
                tensor_fields = ["title", "content"] if index_name == self.unstructured_text_index_name else None
                res = self.client.index(index_name).add_documents(documents, tensor_fields=tensor_fields)
                self.assertFalse(res["errors"])
                
                for search_method, hybrid_params, query_types, description in test_cases:
                    with self.subTest(search_method=search_method, description=description):
                        
                        # Test different query types
                        for query_type in query_types:
                            with self.subTest(query_type=query_type):
                                
                                if query_type == "string":
                                    # Test normal string query
                                    normal_query = "sample text"
                                    base64_query = base64_image
                                    
                                    queries_to_test = [
                                        (normal_query, normal_query, "normal_string"),
                                        (base64_query, "data:image/[omitted]", "base64_string")
                                    ]
                                    
                                elif query_type == "dict":
                                    # Test dict queries
                                    normal_dict_query = {"sample": 0.7, "text": 0.3}
                                    base64_dict_query = {base64_image: 0.8, "text": 0.2}
                                    mixed_dict_query = {base64_image: 0.5, "sample": 0.3, "text": 0.2}
                                    
                                    queries_to_test = [
                                        (normal_dict_query, normal_dict_query, "normal_dict"),
                                        (base64_dict_query, {"data:image/[omitted]": 0.8, "text": 0.2}, "base64_dict"),
                                        (mixed_dict_query, {"data:image/[omitted]": 0.5, "sample": 0.3, "text": 0.2}, "mixed_dict")
                                    ]
                                    
                                
                                # Test each query
                                for input_query, expected_query, query_desc in queries_to_test:
                                    with self.subTest(query_desc=query_desc):
                                        # Skip base64 queries for lexical search (images not supported)
                                        if search_method == "LEXICAL" and query_desc.startswith("base64"):
                                            continue
                                        
                                        # Prepare search parameters
                                        search_params = {
                                            "q": input_query,
                                            "search_method": search_method,
                                            "limit": 5
                                        }
                                        
                                        if hybrid_params:
                                            search_params["hybrid_parameters"] = hybrid_params
                                        
                                        # Execute search
                                        result = self.client.index(index_name).search(**search_params)

                                        # Verify query field matches expected value
                                        self.assertEqual(result["query"], expected_query,
                                                       f"Query field mismatch for {search_method} {query_desc}")

    def test_context_documents_only_lexical_fails(self):
        """
        Tests that Lexical search or hybrid search with lexical/lexical cannot have context docs.
        Test context documents validation with different search methods via API.
        """
        # Test documents
        docs = [
            {"_id": "doc1", "title": "red apple fruit", "content": "sweet taste"},
            {"_id": "doc2", "title": "green apple fruit", "content": "sour taste"},
            {"_id": "context_doc", "title": "apple context", "content": "fruit context"}
        ]
        
        # Context for testing
        context = {
            "documents": {
                "ids": {"context_doc": 1.0},
                "parameters": {
                    "tensorFields": ["title"],
                    "excludeInputDocuments": True
                }
            }
        }
        
        # Test both index types
        for index_name in [self.structured_text_index_name, self.unstructured_text_index_name]:
            with self.subTest(index=index_name):
                # Add documents
                tensor_fields = ["title", "content"] if index_name == self.unstructured_text_index_name else None
                res = self.client.index(index_name).add_documents(docs, tensor_fields=tensor_fields)
                print(res)
                self.assertFalse(res["errors"])
                
                # Test 1: Lexical search with context.documents should fail
                with self.subTest("Lexical with context fails"):
                    with self.assertRaises(MarqoWebError) as cm:
                        self.client.index(index_name).search(
                            q="apple",
                            search_method="LEXICAL",
                            context=context
                        )
                    self.assertEqual(cm.exception.status_code, 422)
                    self.assertIn("Context is not supported for lexical search", str(cm.exception))
                
                # Test 2: Lexical/Lexical hybrid search with context.documents should fail
                with self.subTest("Hybrid lexical/lexical with context fails"):
                    with self.assertRaises(MarqoWebError) as cm:
                        self.client.index(index_name).search(
                            q="apple",
                            search_method="HYBRID",
                            hybrid_parameters={
                                "retrievalMethod": "lexical",
                                "rankingMethod": "lexical"
                            },
                            context=context
                        )
                    self.assertEqual(cm.exception.status_code, 422)
                    self.assertIn("Context is not supported for lexical/lexical hybrid search", str(cm.exception))
                
                # Test 3: Tensor search with context.documents should work
                with self.subTest("Tensor with context works"):
                    result = self.client.index(index_name).search(
                        q={"apple": 1.0},  # Use dict format for tensor search with context
                        search_method="TENSOR",
                        context=context
                    )
                    self.assertIsNotNone(result)
                    self.assertIn("hits", result)
                
                # Test 4: Disjunction/RRF hybrid search with context.documents should work
                with self.subTest("Hybrid disjunction/RRF with context works"):
                    result = self.client.index(index_name).search(
                        q=None,
                        search_method="HYBRID",
                        hybrid_parameters={
                            "retrievalMethod": "disjunction",
                            "rankingMethod": "rrf",
                            "queryTensor": {"apple": 1.0},
                            "queryLexical": "apple"
                        },
                        context=context
                    )
                    self.assertIsNotNone(result)
                    self.assertIn("hits", result)

    def test_search_context_allow_missing_documents_true(self):
        """Test search with context and allow_missing_documents=True allows missing context documents"""
        docs = [
            {
                "_id": "1",
                "title": "Red orchid",
                "content": "flower content",
            },
            {
                "_id": "2", 
                "title": "Red rose",
                "content": "flower content",
            },
            {
                "_id": "3",
                "title": "Europe",
                "content": "continent content",
            },
        ]

        for index_name in [self.structured_text_index_name, self.unstructured_text_index_name]:
            for search_method in ["TENSOR", "HYBRID"]:
                hybrid_parameters = {"retrievalMethod": "tensor", "rankingMethod": "tensor"} \
                    if search_method == "HYBRID" else None
                with self.subTest(f"index_name={index_name}, search_method={search_method}"):
                    tensor_fields = ["title", "content"] if index_name == self.unstructured_text_index_name else None
                    add_docs_results = self.client.index(index_name).add_documents(docs, tensor_fields=tensor_fields)

                    if add_docs_results["errors"]:
                        raise Exception(f"Failed to add documents to index {index_name}")

                    # Should succeed even with missing context document "missing_doc"
                    context = {
                        "documents": {
                            "ids": {"1": 1.0, "2": 1.0, "missing_doc": 1.0},
                            "parameters": {
                                "tensorFields": ["title"],
                                "excludeInputDocuments": True,
                                "allowMissingDocuments": True
                            }
                        }
                    }

                    res = self.client.index(index_name).search(
                        q=None,
                        context=context,
                        search_method=search_method,
                        hybrid_parameters=hybrid_parameters
                    )

                    # Should return results based on available context documents
                    ids = [doc["_id"] for doc in res["hits"]]
                    self.assertIn("3", ids)

    def test_search_context_allow_missing_embeddings_true(self):
        """Test search with context and allow_missing_embeddings=True allows context documents without embeddings"""
        docs = [
            {
                "_id": "1",
                "title": "Red orchid",
            },
            {
                "_id": "2",
                "title": "Red rose", 
                "content": "flower content",
            },
            {
                "_id": "3",
                "title": "Europe",
                "content": "continent content",
            },
        ]

        for index_name in [self.structured_text_index_name, self.unstructured_text_index_name]:
            for search_method in ["TENSOR", "HYBRID"]:
                hybrid_parameters = {"retrievalMethod": "tensor", "rankingMethod": "tensor"} \
                    if search_method == "HYBRID" else None
                with self.subTest(f"index_name={index_name}, search_method={search_method}"):
                    tensor_fields = ["content", "title"] if index_name == self.unstructured_text_index_name else None
                    add_docs_results = self.client.index(index_name).add_documents(docs, tensor_fields=tensor_fields)

                    if add_docs_results["errors"]:
                        raise Exception(f"Failed to add documents to index {index_name}")

                    # Should succeed even when context documents 1 and 2 lack embeddings for title field
                    context = {
                        "documents": {
                            "ids": {"1": 1.0, "2": 1.0},
                            "parameters": {
                                # doc '1' does not have content
                                "tensorFields": ["content"],
                                "excludeInputDocuments": True,
                                "allowMissingEmbeddings": True
                            }
                        }
                    }

                    res = self.client.index(index_name).search(
                        q=None,
                        context=context,
                        search_method=search_method,
                        hybrid_parameters=hybrid_parameters
                    )

                    self.assertEqual("3", res["hits"][0]["_id"])

    def test_search_context_allow_missing_both_true(self):
        """Test search with context and both allow_missing_documents=True and allow_missing_embeddings=True"""
        docs = [
            {
                "_id": "1",
                "content": "flower content",
            },
            {
                "_id": "2",
                "title": "Red rose",
                "content": "flower content",
            },
            {
                "_id": "3", 
                "title": "Europe",
                "content": "continent content",
            },
        ]

        for index_name in [self.structured_text_index_name, self.unstructured_text_index_name]:
            for search_method in ["TENSOR", "HYBRID"]:
                hybrid_parameters = {"retrievalMethod": "tensor", "rankingMethod": "tensor"} \
                    if search_method == "HYBRID" else None
                with self.subTest(f"index_name={index_name}, search_method={search_method}"):
                    tensor_fields = ["content", "title"] if index_name == self.unstructured_text_index_name else None
                    add_docs_results = self.client.index(index_name).add_documents(docs, tensor_fields=tensor_fields)

                    if add_docs_results["errors"]:
                        raise Exception(f"Failed to add documents to index {index_name}")

                    # Should succeed with both missing context documents and missing embeddings
                    context = {
                        "documents": {
                            # doc '1' has no title embeddings
                            "ids": {"1": 1.0, "2": 1.0, "missing_doc": 1.0},
                            "parameters": {
                                "tensorFields": ["title"],
                                "excludeInputDocuments": True,
                                "allowMissingDocuments": True,
                                "allowMissingEmbeddings": True
                            }
                        }
                    }

                    res = self.client.index(index_name).search(
                        q=None,
                        context=context,
                        search_method=search_method,
                        hybrid_parameters=hybrid_parameters
                    )

                    self.assertEqual("3", res["hits"][0]["_id"])

    def test_search_context_failed_to_collect_vectors_error(self):
        """Test search with context and both allow_missing_documents=True and allow_missing_embeddings=True"""
        docs = [
            {
                "_id": "1",
                "content": "flower content",
            },
            {
                "_id": "2",
                "title": "Red rose",
                "content": "flower content",
            },
            {
                "_id": "3",
                "title": "Europe",
                "content": "continent content",
            },
        ]

        for index_name in [self.structured_text_index_name, self.unstructured_text_index_name]:
            for search_method in ["TENSOR", "HYBRID"]:
                hybrid_parameters = {"retrievalMethod": "tensor", "rankingMethod": "tensor"} \
                    if search_method == "HYBRID" else None
                with self.subTest(f"index_name={index_name}, search_method={search_method}"):
                    tensor_fields = ["content", "title"] if index_name == self.unstructured_text_index_name else None
                    add_docs_results = self.client.index(index_name).add_documents(docs, tensor_fields=tensor_fields)

                    if add_docs_results["errors"]:
                        raise Exception(f"Failed to add documents to index {index_name}")

                    # Should succeed with both missing context documents and missing embeddings
                    context = {
                        "documents": {
                            # doc '1' has no title embeddings
                            "ids": {"1": 1.0, "missing_doc": 1.0},
                            "parameters": {
                                "tensorFields": ["title"],
                                "excludeInputDocuments": True,
                                "allowMissingDocuments": True,
                                "allowMissingEmbeddings": True
                            }
                        }
                    }
                    with self.assertRaises(MarqoWebError) as e:
                        _ = self.client.index(index_name).search(
                            q=None,
                            context=context,
                            search_method=search_method,
                            hybrid_parameters=hybrid_parameters
                        )
                    self.assertIn("Marqo could not collect any vectors from the search query", str(e.exception))