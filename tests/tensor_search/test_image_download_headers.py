"""
Module for testing image download headers.
"""
import unittest.mock
import os
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
# we are renaming get to prevent inf. recursion while mocking get():
from requests import get as requests_get
from marqo.tensor_search.models.api_models import BulkSearchQuery
from unittest import mock
import requests
from marqo.s2_inference.clip_utils import load_image_from_path
from marqo.tensor_search.enums import IndexSettingsField
from marqo.errors import IndexNotFoundError
from marqo.tensor_search import tensor_search
from tests.marqo_test import MarqoTestCase


class TestImageDownloadHeaders(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super(TestImageDownloadHeaders, cls).setUpClass()
        cls.generic_header = {"Content-type": "application/json"}
        cls.index_name_1 = "my-test-index-1"
        cls.real_img_url = 'https://github.com/marqo-ai/marqo-api-tests/blob/mainline/assets/ai_hippo_realistic.png'

    def setUp(self) -> None:
        self.endpoint = self.authorized_url
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError:
            pass
        
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.index_name_1 = "my-test-index-1"
        try:
            tensor_search.delete_index(config=cls.config, index_name=cls.index_name_1)
        except IndexNotFoundError:
            pass
        
    def tearDown(self):
        self.device_patcher.stop()

    def image_index_settings(self) -> dict:
        return {
                IndexSettingsField.index_defaults: {
                    IndexSettingsField.model: "ViT-B/32",
                    IndexSettingsField.treat_urls_and_pointers_as_images: True,
                }
            }

    def test_img_download_search(self):
        # Create a vector index and add a document with an image URL
        tensor_search.create_vector_index(
            config=self.config, index_name=self.index_name_1, index_settings=self.image_index_settings()
        )
        image_download_headers = {"Authorization": "some secret key blah"}
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[
                {"_id": "1", "image": self.real_img_url}],
            auto_refresh=True, image_download_headers=image_download_headers, device="cpu"))

        def pass_through_requests_get(url, *args, **kwargs):
            return requests_get(url, *args, **kwargs)

        mock_get = unittest.mock.MagicMock()
        mock_get.side_effect = pass_through_requests_get

        # Mock the requests.get method to check if the headers are passed correctly
        with mock.patch("requests.get", mock_get):
            with mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {mock_get, requests.post, requests.put}):
                # Perform a vector search
                search_res = tensor_search._vector_text_search(
                    config=self.config, index_name=self.index_name_1,
                    result_count=1, query=self.real_img_url, image_download_headers=image_download_headers, device="cpu"
                )
                # Check if the image URL was called at least once with the correct headers
                image_url_called = any(
                    call_args[0] == self.real_img_url and call_kwargs.get('headers', None) == image_download_headers
                    for call_args, call_kwargs in mock_get.call_args_list
                )
                assert image_url_called, "Image URL not called with the correct headers"

    def test_img_download_add_docs(self):

        tensor_search.create_vector_index(
            index_name=self.index_name_1, config=self.config, index_settings=self.image_index_settings())

        def pass_through_load_image_from_path(*arg, **kwargs):
            return load_image_from_path(*arg, **kwargs)

        mock_load_image_from_path = unittest.mock.MagicMock()
        mock_load_image_from_path.side_effect = pass_through_load_image_from_path

        @unittest.mock.patch("marqo.s2_inference.clip_utils.load_image_from_path", mock_load_image_from_path)
        def run():

            image_download_headers = {"Authorization": "some secret key blah"}

            # Add a document with an image URL
            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, docs=[
                    { "_id": "1", "image": self.real_img_url}
                ], auto_refresh=True, image_download_headers=image_download_headers, device="cpu"
            ))
            # Check if load_image_from_path was called with the correct headers
            assert len(mock_load_image_from_path.call_args_list) == 1
            call_args, call_kwargs = mock_load_image_from_path.call_args_list[0]
            assert image_download_headers in call_args
            return True

        assert run() is True

    def test_img_download_bulk_search(self):
        # Create a vector index and add a document with an image URL
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1,
                                          index_settings=self.image_index_settings())
        test_image_url = self.real_img_url
        image_download_headers = {"Authorization": "some secret key blah"}

        def pass_through_load_image_from_path(*args, **kwargs):
            return load_image_from_path(*args, **kwargs)

        def pass_through_requests_get(url, *args, **kwargs):
            if url == test_image_url:
                assert kwargs.get('headers', None) == image_download_headers
            return requests_get(url, *args, **kwargs)

        # Mock the load_image_from_path function
        mock_load_image_from_path = unittest.mock.MagicMock()
        mock_load_image_from_path.side_effect = pass_through_load_image_from_path

        with unittest.mock.patch("marqo.s2_inference.clip_utils.load_image_from_path", mock_load_image_from_path):
            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, docs=[
                {
                    "_id": "1",
                    "image": test_image_url,
                }],
                auto_refresh=True, image_download_headers=image_download_headers, device="cpu"))

        # Set up the mock GET
        mock_get = unittest.mock.MagicMock()
        mock_get.side_effect = pass_through_requests_get

        with mock.patch("requests.get", mock_get):
            with mock.patch('marqo._httprequests.ALLOWED_OPERATIONS', {mock_get, requests.post, requests.put}):
                bulk_search_query = BulkSearchQuery(queries=[{
                    "index": self.index_name_1,
                    "q": self.real_img_url,
                    "image_download_headers": image_download_headers
                }])
                resp = tensor_search.bulk_search(marqo_config=self.config, query=bulk_search_query)

        # Check if the image URL was called at least once with the correct headers
        image_url_called = any(
            call_args[0] == test_image_url and call_kwargs.get('headers', None) == image_download_headers
            for call_args, call_kwargs in mock_get.call_args_list
        )
        assert image_url_called, "Image URL not called with the correct headers"







