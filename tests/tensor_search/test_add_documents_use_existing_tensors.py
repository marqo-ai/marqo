import os
import unittest
from unittest import mock

from marqo.core.models.marqo_index import FieldType
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase


class TestAddDocumentsUseExistingTensors(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        structured_index = cls.structured_marqo_index_request(
            fields=[FieldRequest(name="text_field_1", type="text"),
                    FieldRequest(name="text_field_2", type="text")],
            tensor_fields=["text_field_1", "text_field_2"]
        )
        structured_index_multimodal = cls.structured_marqo_index_request(
            fields=[
                FieldRequest(name="text_field_1", type="text"),
                FieldRequest(name="text_field_2", type="text"),
                FieldRequest(name='multimodal_field', type=FieldType.MultimodalCombination,
                             dependent_fields={'text_field_1': 0.5, 'text_field_2': 0.8})
            ],
            tensor_fields=["multimodal_field"]
        )

        unstructured_index = cls.unstructured_marqo_index_request()

        cls.indexes = cls.create_indexes([
            structured_index,
            structured_index_multimodal,
            unstructured_index
        ])

        cls.structured_index = structured_index.name
        cls.structured_index_multimodal = structured_index_multimodal.name
        cls.unstructured_index = unstructured_index.name

    def setUp(self) -> None:
        super().setUp()
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def test_use_existing_tensor_no_change(self):
        """
        Checks that the vectors are not updated if the content is the same
        """
        doc = {
            "text_field_1": "content 1",
            "_id": "1"
        }

        from marqo.s2_inference import s2_inference
        original_vectorise = s2_inference.vectorise

        for index_name in [self.structured_index, self.unstructured_index]:
            tensor_fields = None if index_name == self.structured_index else ["text_field_1", "text_field_2"]
            with self.subTest(f"{index_name}"):
                with mock.patch.object(s2_inference,
                                       'vectorise',
                                       side_effect=original_vectorise) as mock_vectorise:
                    tensor_search.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index_name,
                            docs=[doc],
                            tensor_fields=tensor_fields,
                        )
                    )

                    self.assertEqual(1, mock_vectorise.call_count, 'vectorise should be called once')

                with mock.patch.object(s2_inference,
                                       'vectorise',
                                       side_effect=original_vectorise) as mock_vectorise:
                    tensor_search.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index_name,
                            docs=[doc],
                            tensor_fields=tensor_fields,
                            use_existing_tensors=True
                        )
                    )

                    self.assertEqual(0, mock_vectorise.call_count, 'vectorise should not be called')

                    search_res = tensor_search.search(config=self.config, index_name=index_name, text="content")
                    get_doc_res = tensor_search.get_document_by_id(config=self.config, index_name=index_name,
                                                                   document_id="1", show_vectors=True)

                    self.assertEqual("content 1", search_res["hits"][0]["text_field_1"])
                    self.assertEqual(1, len(get_doc_res["_tensor_facets"]))
                    self.assertEqual("content 1", get_doc_res["_tensor_facets"][0]["text_field_1"])

    def test_use_existing_tensor_new_fields(self):
        doc_1 = {
            "text_field_1": "content 1",
            "_id": "1"
        }

        doc_2 = {
            "text_field_2": "content 2",
            "_id": "1"
        }

        from marqo.s2_inference import s2_inference
        original_vectorise = s2_inference.vectorise

        for index_name in [self.structured_index, self.unstructured_index]:
            tensor_fields = None if index_name == self.structured_index else ["text_field_1", "text_field_2"]
            with self.subTest(f"{index_name}"):
                with mock.patch.object(s2_inference,
                                       'vectorise',
                                       side_effect=original_vectorise) as mock_vectorise:
                    tensor_search.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index_name,
                            docs=[doc_1],
                            tensor_fields=tensor_fields,
                        )
                    )

                    self.assertEqual(1, mock_vectorise.call_count, 'vectorise should be called once')

                with mock.patch.object(s2_inference,
                                       'vectorise',
                                       side_effect=original_vectorise) as mock_vectorise:
                    tensor_search.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index_name,
                            docs=[doc_2],
                            tensor_fields=tensor_fields,
                            use_existing_tensors=True
                        )
                    )

                    self.assertEqual(1, mock_vectorise.call_count, 'vectorise should not be called')

                    search_res = tensor_search.search(config=self.config, index_name=index_name, text="content")
                    get_doc_res = tensor_search.get_document_by_id(config=self.config, index_name=index_name,
                                                                   document_id="1", show_vectors=True)

                    self.assertEqual("content 2", search_res["hits"][0]["text_field_2"])
                    self.assertEqual(1, len(get_doc_res["_tensor_facets"]))
                    self.assertEqual("content 2", get_doc_res["_tensor_facets"][0]["text_field_2"])

    def test_use_existing_tensor_multimodal_no_change(self):
        """
        Checks that the vectors are not updated if the content is the same for multimodal fields
        """
        doc = {
            "text_field_1": "content 1",
            "text_field_2": "content 2",
            "_id": "1"
        }

        from marqo.s2_inference import s2_inference
        original_vectorise = s2_inference.vectorise

        for index_name in [self.structured_index_multimodal, self.unstructured_index]:
            tensor_fields = None if index_name == self.structured_index_multimodal else ["multimodal_field"]
            mappings = None if index_name == self.structured_index else \
                {
                    "multimodal_field": {"type": "multimodal_combination", "weights": {
                        "text_field_1": 0.5, "text_field_2": 0.8}}
                }
            with self.subTest(f"{index_name}"):
                with mock.patch.object(s2_inference,
                                       'vectorise',
                                       side_effect=original_vectorise) as mock_vectorise:
                    tensor_search.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index_name,
                            docs=[doc],
                            tensor_fields=tensor_fields,
                            mappings=mappings
                        )
                    )

                    self.assertEqual(1, mock_vectorise.call_count, 'vectorise should be called once')

                with mock.patch.object(s2_inference,
                                       'vectorise',
                                       side_effect=original_vectorise) as mock_vectorise:
                    tensor_search.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index_name,
                            docs=[doc],
                            tensor_fields=tensor_fields,
                            mappings=mappings,
                            use_existing_tensors=True
                        )
                    )

                    self.assertEqual(0, mock_vectorise.call_count, 'vectorise should not be called')

                    search_res = tensor_search.search(config=self.config, index_name=index_name, text="content")
                    get_doc_res = tensor_search.get_document_by_id(config=self.config, index_name=index_name,
                                                                   document_id="1", show_vectors=True)

                    self.assertEqual("content 1", search_res["hits"][0]["text_field_1"])
                    self.assertEqual("content 2", search_res["hits"][0]["text_field_2"])
                    self.assertEqual(1, len(get_doc_res["_tensor_facets"]))
                    self.assertEqual("content 1", get_doc_res["_tensor_facets"][0]["text_field_1"])

    def test_use_existing_tensor_multimodal_added(self):
        """
        Checks that the vectors are updated if a multimodal field is added
        """
        doc_1 = {
            "_id": "1"
        }
        doc_2 = {
            "text_field_1": "content 1",
            "text_field_2": "content 2",
            "_id": "1"
        }

        from marqo.s2_inference import s2_inference
        original_vectorise = s2_inference.vectorise

        for index_name in [self.structured_index_multimodal, self.unstructured_index]:
            tensor_fields_1 = None if index_name == self.structured_index_multimodal else []
            tensor_fields_2 = None if index_name == self.structured_index_multimodal else ["multimodal_field"]
            mappings = None if index_name == self.structured_index else \
                {
                    "multimodal_field": {"type": "multimodal_combination", "weights": {
                        "text_field_1": 0.5, "text_field_2": 0.8}}
                }
            with self.subTest(f"{index_name}"):
                with mock.patch.object(s2_inference,
                                       'vectorise',
                                       side_effect=original_vectorise) as mock_vectorise:
                    tensor_search.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index_name,
                            docs=[doc_1],
                            tensor_fields=tensor_fields_1,  # No tensor fields
                            mappings=None
                        )
                    )

                    self.assertEqual(1, mock_vectorise.call_count, 'vectorise should be called once')

                with mock.patch.object(s2_inference,
                                       'vectorise',
                                       side_effect=original_vectorise) as mock_vectorise:
                    tensor_search.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index_name,
                            docs=[doc_2],
                            tensor_fields=tensor_fields_2,
                            mappings=mappings,
                            use_existing_tensors=True
                        )
                    )

                    self.assertEqual(0, mock_vectorise.call_count, 'vectorise should not be called')

                    search_res = tensor_search.search(config=self.config, index_name=index_name, text="content")
                    get_doc_res = tensor_search.get_document_by_id(config=self.config, index_name=index_name,
                                                                   document_id="1", show_vectors=True)

                    self.assertEqual("content 1", search_res["hits"][0]["text_field_1"])
                    self.assertEqual("content 2", search_res["hits"][0]["text_field_2"])
                    self.assertEqual(1, len(get_doc_res["_tensor_facets"]))
                    self.assertEqual("content 1", get_doc_res["_tensor_facets"][0]["text_field_1"])

    def test_use_existing_tensor_multimodal_changed(self):
        """
        Checks that the vectors are updated if a multimodal field is added
        """
        doc_1 = {
            "text_field_1": "content 1",
            "text_field_2": "content 2",
            "_id": "1"
        }
        doc_2 = {
            "text_field_1": "content 1",
            "text_field_2": "content 2-updated",
            "_id": "1"
        }

        from marqo.s2_inference import s2_inference
        original_vectorise = s2_inference.vectorise

        for index_name in [self.structured_index_multimodal, self.unstructured_index]:
            tensor_fields = None if index_name == self.structured_index_multimodal else ["multimodal_field"]
            mappings = None if index_name == self.structured_index else \
                {
                    "multimodal_field": {"type": "multimodal_combination", "weights": {
                        "text_field_1": 0.5, "text_field_2": 0.8}}
                }
            with self.subTest(f"{index_name}"):
                with mock.patch.object(s2_inference,
                                       'vectorise',
                                       side_effect=original_vectorise) as mock_vectorise:
                    tensor_search.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index_name,
                            docs=[doc_1],
                            tensor_fields=tensor_fields,
                            mappings=mappings
                        )
                    )

                    self.assertEqual(1, mock_vectorise.call_count, 'vectorise should be called once')

                with mock.patch.object(s2_inference,
                                       'vectorise',
                                       side_effect=original_vectorise) as mock_vectorise:
                    tensor_search.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index_name,
                            docs=[doc_2],
                            tensor_fields=tensor_fields,
                            mappings=mappings,
                            use_existing_tensors=True
                        )
                    )

                    self.assertEqual(1, mock_vectorise.call_count, 'vectorise should be called again')

                    search_res = tensor_search.search(config=self.config, index_name=index_name, text="content")
                    get_doc_res = tensor_search.get_document_by_id(config=self.config, index_name=index_name,
                                                                   document_id="1", show_vectors=True)

                    self.assertEqual("content 1", search_res["hits"][0]["text_field_1"])
                    self.assertEqual("content 2-updated", search_res["hits"][0]["text_field_2"])
                    self.assertEqual(1, len(get_doc_res["_tensor_facets"]))
                    self.assertEqual('{"text_field_1": "content 1", "text_field_2": "content 2-updated"}',
                                     get_doc_res["_tensor_facets"][0]["multimodal_field"])

    @unittest.skip
    def test_use_existing_tensors_resilience(self):
        """should if one doc fails validation, the rest should still be inserted
        """
        d1 = {
            "title 1": "content 1",
            "desc 2": "content 2. blah blah blah"
        }
        # 1 valid ID doc:
        res = tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1,
                                                              docs=[d1, {'_id': 1224}, {"_id": "fork", "abc": "123"}],
                                                              auto_refresh=True, use_existing_tensors=True,
                                                              device="cpu"))
        assert [item['status'] for item in res['items']] == [201, 400, 201]

        # no valid IDs
        res_no_valid_id = tensor_search.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(index_name=self.index_name_1, docs=[d1, {'_id': 1224}, d1],
                                          auto_refresh=True, use_existing_tensors=True, device="cpu"))
        # we also should not be send in a get request as there are no valid document IDs
        assert [item['status'] for item in res_no_valid_id['items']] == [201, 400, 201]

    @unittest.skip
    def test_use_existing_tensors_no_id(self):
        """should insert if there's no ID
        """
        d1 = {
            "title 1": "content 1",
            "desc 2": "content 2. blah blah blah"
        }
        r1 = tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1, docs=[d1],
                                                              auto_refresh=True, use_existing_tensors=True,
                                                              device="cpu"))
        r2 = tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1, docs=[d1, d1],
                                                              auto_refresh=True, use_existing_tensors=True,
                                                              device="cpu"))

        for item in r1['items']:
            assert item['result'] == 'created'
        for item in r2['items']:
            assert item['result'] == 'created'

    @unittest.skip
    def test_use_existing_tensors_non_existing(self):
        """check parity between a doc created with and without use_existing_tensors, then overwritten,
        for a newly created doc.
        """
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[
                {
                    "_id": "123",
                    "title 1": "content 1",
                    "desc 2": "content 2. blah blah blah"
                }], auto_refresh=True, use_existing_tensors=False, device="cpu"))

        regular_doc = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="123", show_vectors=True)

        tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)

        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[
                {
                    "_id": "123",
                    "title 1": "content 1",
                    "desc 2": "content 2. blah blah blah"
                }], auto_refresh=True, use_existing_tensors=True, device="cpu"))
        use_existing_tensors_doc = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="123", show_vectors=True)
        self.assertEqual(use_existing_tensors_doc, regular_doc)

        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[
                {
                    "_id": "123",
                    "title 1": "content 1",
                    "desc 2": "content 2. blah blah blah"
                }], auto_refresh=True, use_existing_tensors=True, device="cpu"))
        overwritten_doc = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="123", show_vectors=True)

        self.assertEqual(use_existing_tensors_doc, overwritten_doc)

    @unittest.skip
    def test_use_existing_tensors_dupe_ids(self):
        """
        Should only use the latest inserted ID. Make sure it doesn't get the first/middle one
        """

        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[
                {
                    "_id": "3",
                    "title": "doc 3b"
                },

            ], auto_refresh=True, device="cpu"))

        doc_3_solo = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="3", show_vectors=True)

        tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[
                {
                    "_id": "1",
                    "title": "doc 1"
                },
                {
                    "_id": "2",
                    "title": "doc 2",
                },
                {
                    "_id": "3",
                    "title": "doc 3a",
                },
                {
                    "_id": "3",
                    "title": "doc 3b"
                }],
            auto_refresh=True, use_existing_tensors=True, device="cpu"))

        doc_3_duped = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="3", show_vectors=True)

        self.assertEqual(doc_3_solo, doc_3_duped)

        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[
                {
                    "_id": "1",
                    "title": "doc 1"
                },
                {
                    "_id": "2",
                    "title": "doc 2",
                },
                {
                    "_id": "3",
                    "title": "doc 3a",
                },
                {
                    "_id": "3",
                    "title": "doc 3b"
                },

            ], auto_refresh=True, use_existing_tensors=True, device="cpu"))

        doc_3_overwritten = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="3", show_vectors=True)

        # Needs to be 3b, not 3a
        self.assertEqual(doc_3_duped, doc_3_overwritten)

    @unittest.skip
    def test_use_existing_tensors_retensorize_fields(self):
        """
        During the initial index, some fields are non-tensor fields
        When we insert the doc again, with use_existing_tensors, we make them tensor fields.
        They should still have no tensors.
        """

        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[
                {
                    "_id": "123",
                    "title 1": "content 1",
                    "title 2": 2,
                    "title 3": True,
                    "title 4": "content 4"
                }], auto_refresh=True, use_existing_tensors=True,
            non_tensor_fields=["title 1", "title 2", "title 3", "title 4"], device="cpu"))
        d1 = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="123", show_vectors=True)
        assert len(d1["_tensor_facets"]) == 0

        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[
                {
                    "_id": "123",
                    "title 1": "content 1",
                    "title 2": 2,
                    "title 3": True,
                    "title 4": "content 4"
                }], auto_refresh=True, use_existing_tensors=True, device="cpu"))
        d2 = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="123", show_vectors=True)

        assert len(d2["_tensor_facets"]) == 0

    @unittest.skip
    def test_use_existing_tensors_getting_non_tensorised(self):
        """
        During the initial index, one field is set as a non_tensor_field.
        When we insert the doc again, with use_existing_tensors, because the content
        hasn't changed, we use the existing (non-existent) vectors
        """
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[
                {
                    "_id": "123",
                    "title 1": "content 1",
                    "non-tensor-field": "content 2. blah blah blah"
                }], auto_refresh=True, non_tensor_fields=["non-tensor-field"], device="cpu"))
        d1 = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="123", show_vectors=True)
        assert len(d1["_tensor_facets"]) == 1
        assert "title 1" in d1["_tensor_facets"][0]

        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[
                {
                    "_id": "123",
                    "title 1": "content 1",
                    "non-tensor-field": "content 2. blah blah blah"
                }], auto_refresh=True, use_existing_tensors=True, device="cpu"))
        d2 = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="123", show_vectors=True)
        self.assertEqual(d1["_tensor_facets"], d2["_tensor_facets"])

        # The only field is a non-tensor field. This makes a chunkless doc.
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[
                {
                    "_id": "999",
                    "non-tensor-field": "content 2. blah blah blah"
                }], auto_refresh=True, non_tensor_fields=["non-tensor-field"], device="cpu"))
        d1 = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="999", show_vectors=True)
        assert len(d1["_tensor_facets"]) == 0

        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[
                {
                    "_id": "999",
                    "non-tensor-field": "content 2. blah blah blah"
                }], auto_refresh=True, use_existing_tensors=True, device="cpu"))
        d2 = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="999", show_vectors=True)
        self.assertEqual(d1["_tensor_facets"], d2["_tensor_facets"])

    @unittest.skip
    def test_use_existing_tensors_check_updates(self):
        """ Check to see if the document has been appropriately updated
        """
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[
                {
                    "_id": "123",
                    "title 1": "content 1",
                    "modded field": "original content",
                    "non-tensor-field": "content 2. blah blah blah"
                }], auto_refresh=True, non_tensor_fields=["non-tensor-field"], device="cpu"))

        def pass_through_vectorise(*arg, **kwargs):
            """Vectorise will behave as usual, but we will be able to see the call list
            via mock
            """
            return vectorise(*arg, **kwargs)

        mock_vectorise = unittest.mock.MagicMock()
        mock_vectorise.side_effect = pass_through_vectorise

        @unittest.mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, docs=[
                    {
                        "_id": "123",
                        "title 1": "content 1",  # this one should keep the same vectors
                        "my new field": "cat on mat",  # new vectors because it's a new field
                        "modded field": "updated content",  # new vectors because the content is modified
                        "non-tensor-field": "content 2. blah blah blah",  # this would should still have no vectors
                        "2nd-non-tensor-field": "content 2. blah blah blah"
                        # this one is explicitly being non-tensorised
                    }], auto_refresh=True, non_tensor_fields=["2nd-non-tensor-field"], use_existing_tensors=True,
                device="cpu"))
            content_to_be_vectorised = [call_kwargs['content'] for call_args, call_kwargs
                                        in mock_vectorise.call_args_list]
            assert content_to_be_vectorised == [["cat on mat"], ["updated content"]]
            return True

        assert run()

    @unittest.skip
    def test_use_existing_tensors_check_meta_data(self):
        """

        Checks chunk meta data and vectors are as expected

        """
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[
                {
                    "_id": "123",
                    "title 1": "content 1",
                    "modded field": "original content",
                    "non-tensor-field": "content 2. blah blah blah",
                    "field_that_will_disappear": "some stuff",  # this gets dropped during the next add docs call,
                    "field_to_be_list": "some stuff",
                    "fl": 1.51
                }], auto_refresh=True, non_tensor_fields=["non-tensor-field"], device="cpu"))

        use_existing_tensor_doc = {
            "title 1": "content 1",  # this one should keep the same vectors
            "my new field": "cat on mat",  # new vectors because it's a new field
            "modded field": "updated content",  # new vectors because the content is modified
            "non-tensor-field": "content 2. blah blah blah",  # this would should still have no vectors
            "2nd-non-tensor-field": "content 2. blah blah blah",  # this one is explicitly being non-tensorised,
            # should end up in meta data:
            "field_to_be_list": ["hi", "there"],
            "new_field_list": ["some new content"],
            "fl": 101.3,
            "new_bool": False
        }
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, docs=[{"_id": "123", **use_existing_tensor_doc}],
                auto_refresh=True, non_tensor_fields=["2nd-non-tensor-field", "field_to_be_list", 'new_field_list'],
                use_existing_tensors=True, device="cpu"))

        updated_doc = requests.get(
            url=F"{self.endpoint}/{self.index_name_1}/_doc/123",
            verify=False
        )
        chunks = [chunk for chunk in updated_doc.json()['_source']['__chunks']]
        # each chunk needs its metadata to be the same as the updated document's content
        for ch in chunks:
            ch_meta_data = {k: v for k, v in ch.items() if not k.startswith("__")}
            assert use_existing_tensor_doc == ch_meta_data
        assert len(chunks) == 3

        # check if the vectors/field content is correct
        for vector_field in ["title 1", "my new field", "modded field"]:
            found_vector_field = False
            for ch in chunks:
                if ch["__field_name"] == vector_field:
                    found_vector_field = True
                    assert ch['__field_content'] == use_existing_tensor_doc[vector_field]
                assert isinstance(ch[TensorField.marqo_knn_field], list)
            assert found_vector_field

    @unittest.skip
    def test_use_existing_tensors_check_meta_data_mappings(self):
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[
                {
                    "_id": "123",
                    "title 1": "content 1",
                    "modded field": "original content",
                    "non-tensor-field": "content 2. blah blah blah",
                    "field_that_will_disappear": "some stuff",  # this gets dropped during the next add docs call
                    "field_to_be_list": "some stuff",
                    "fl": 1.51
                }], auto_refresh=True, non_tensor_fields=["non-tensor-field"], device="cpu"))

        use_existing_tensor_doc = {
            "title 1": "content 1",  # this one should keep the same vectors
            "my new field": "cat on mat",  # new vectors because it's a new field
            "modded field": "updated content",  # new vectors because the content is modified
            "non-tensor-field": "content 2. blah blah blah",  # this would should still have no vectors
            "2nd-non-tensor-field": "content 2. blah blah blah",  # this one is explicitly being non-tensorised,
            # should end up in meta data:
            "field_to_be_list": ["hi", "there"],
            "new_field_list": ["some new content"],
            "fl": 101.3,
            "new_bool": False
        }
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1,
                                                              docs=[{"_id": "123", **use_existing_tensor_doc}],
                                                              auto_refresh=True,
                                                              non_tensor_fields=["2nd-non-tensor-field",
                                                                                 "field_to_be_list", 'new_field_list'],
                                                              use_existing_tensors=True, device="cpu"))

        tensor_search.index_meta_cache.refresh_index(config=self.config, index_name=self.index_name_1)

        index_info = tensor_search.get_index_info(config=self.config, index_name=self.index_name_1)
        # text or list of texts:
        text_fields = ["title 1", "my new field", "modded field", "non-tensor-field", "2nd-non-tensor-field",
                       "field_to_be_list", "new_field_list", "field_that_will_disappear"]

        for text_field in text_fields:
            assert index_info.properties[text_field]['type'] == 'text'
            assert index_info.properties['__chunks']['properties'][text_field]['type'] == 'keyword'

        # Only 1 tensor field
        assert index_info.properties['__chunks']['properties'][TensorField.marqo_knn_field]['type'] == 'knn_vector'

        for field_name, os_type in [('fl', "float"), ('new_bool', "boolean")]:
            assert index_info.properties[field_name]['type'] == os_type
            assert index_info.properties['__chunks']['properties'][field_name]['type'] == os_type

    @unittest.skip
    def test_use_existing_tensors_long_strings_and_images(self):
        """Checks vectorise calls and chunk structure for image and text fields with more than 1 chunk"""
        index_settings = {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": True,
                "model": "ViT-B/32",
                "text_preprocessing": {
                    "split_method": "sentence",
                    "split_length": 1,
                }
            }
        }
        tensor_search.create_vector_index(
            index_name=self.index_name_2, index_settings=index_settings, config=self.config)
        hippo_img = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        artefact_hippo_img = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png'
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_2, docs=[
                {
                    "_id": "123",
                    "txt_to_be_the_same": "some text to leave unchanged. I repeat, unchanged",
                    "txt_field_to_be_deleted": "This is my first sentence. This is my second",
                    "txt_to_be_modified": "this is the original 1st sentence. This is the original 2nd.",
                    "img_to_be_deleted": hippo_img,
                    "img_to_be_modified": hippo_img,
                    "img_to_be_same": hippo_img,
                    "fl": 1.23,
                    "non-tensor-field": ["what", "is", "the", "time"]

                }], auto_refresh=True, non_tensor_fields=["non-tensor-field"], device="cpu"))

        def pass_through_vectorise(*arg, **kwargs):
            """Vectorise will behave as usual, but we will be able to see the call list
            via mock
            """
            return vectorise(*arg, **kwargs)

        mock_vectorise = unittest.mock.MagicMock()
        mock_vectorise.side_effect = pass_through_vectorise
        mock_close = unittest.mock.MagicMock()

        @unittest.mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        @unittest.mock.patch("PIL.Image.Image.close", mock_close)  # disable .close so we can access images in this test
        def run():
            use_existing_tensor_doc = {
                "txt_to_be_the_same": "some text to leave unchanged. I repeat, unchanged",
                "txt_to_be_modified": "this is the updated 1st sentence. This is my second",
                # 2nd sentence not modified
                "txt_to_be_created": "this is a brand new sentence. Yes it is",
                "img_to_be_modified": artefact_hippo_img,
                "img_to_be_same": hippo_img,
                "img_to_be_Created": artefact_hippo_img,
                "fl": 3.5,
                "non-tensor-field": ["it", "is", "9", "o clock"]
            }
            tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_2,
                                                                  docs=[{"_id": "123", **use_existing_tensor_doc}],
                                                                  auto_refresh=True,
                                                                  non_tensor_fields=["non-tensor-field"],
                                                                  use_existing_tensors=True, device="cpu"))

            vectorised_content = [call_kwargs['content'] for call_args, call_kwargs
                                  in mock_vectorise.call_args_list]
            artefact_pil_image = load_image_from_path(artefact_hippo_img, image_download_headers={})
            expected_to_be_vectorised = [
                ["this is the updated 1st sentence.", "This is my second"],
                ["this is a brand new sentence.", "Yes it is"],
                [artefact_pil_image], [artefact_pil_image]]
            assert vectorised_content == expected_to_be_vectorised

            updated_doc = requests.get(
                url=F"{self.endpoint}/{self.index_name_2}/_doc/123",
                verify=False
            )

            parent_doc = updated_doc.json()['_source']
            del parent_doc['__chunks']
            assert parent_doc == use_existing_tensor_doc

            # each chunk needs its metadata to be the same as the updated document's content:
            chunks = [chunk for chunk in updated_doc.json()['_source']['__chunks']]
            for ch in chunks:
                ch_meta_data = {k: v for k, v in ch.items() if not k.startswith("__")}
                assert use_existing_tensor_doc == ch_meta_data

            vector_img_fields = ["img_to_be_modified", "img_to_be_same", "img_to_be_Created"]
            # check if the vectors/field content is correct for images:
            for vector_field in vector_img_fields:
                found_vector_field = False
                for ch in chunks:
                    if ch["__field_name"] == vector_field:
                        found_vector_field = True
                        assert ch['__field_content'] == use_existing_tensor_doc[vector_field]
                    # Only 1 tensor field
                    assert isinstance(ch[TensorField.marqo_knn_field], list)
                assert found_vector_field

            expected_text_chunks = {
                ("txt_to_be_the_same", "some text to leave unchanged."),
                ("txt_to_be_the_same", "I repeat, unchanged"),
                ("txt_to_be_modified", "this is the updated 1st sentence."),
                ("txt_to_be_modified", "This is my second"),
                ("txt_to_be_created", "this is a brand new sentence."),
                ("txt_to_be_created", "Yes it is")
            }
            real_txt_chunks = {(ch["__field_name"], ch["__field_content"])
                               for ch in chunks if ch["__field_name"].startswith("txt")}
            assert real_txt_chunks == expected_text_chunks
            assert len(chunks) == len(vector_img_fields) + len(expected_text_chunks)
            return True

        assert run()

    @unittest.skip
    def test_use_existing_tensors_all_data_types(self):
        """
        Ensure no errors occur even with chunkless docs. (only int, only bool, etc)
        Replacing doc doesn't change the content
        """
        self.maxDiff = None

        doc_args = [
            # Each doc only has 1 type
            [{"_id": "1", "field1": "hello world"},
             {"_id": "2", "field2": True},
             {"_id": "3", "field3": 12345},
             {"_id": "4", "field4": [1, 2, 3]}],

            # Doc with all types
            [{"_id": "1", "field1": "hello world", "field2": True, "field3": 12345, "field4": [1, 2, 3]}],
        ]

        for doc_arg in doc_args:
            # Add doc normally without use_existing_tensors
            add_res = tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1,
                                                                  docs=doc_arg, auto_refresh=True, device="cpu"))

            d1 = tensor_search.get_documents_by_ids(
                config=self.config, index_name=self.index_name_1,
                document_ids=[doc["_id"] for doc in doc_arg], show_vectors=True)

            # Then replace doc with use_existing_tensors
            add_res = tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1,
                                                                  docs=doc_arg, auto_refresh=True,
                                                                  use_existing_tensors=True, device="cpu"))

            d2 = tensor_search.get_documents_by_ids(
                config=self.config, index_name=self.index_name_1,
                document_ids=[doc["_id"] for doc in doc_arg], show_vectors=True)

            self.assertEqual(d1, d2)
