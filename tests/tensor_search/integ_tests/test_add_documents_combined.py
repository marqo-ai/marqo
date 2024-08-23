import os
import uuid
from unittest import mock
from unittest.mock import patch

import PIL
import requests
import torch
from torch import Tensor
from urllib3.exceptions import ProtocolError

from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.s2_inference import types
from marqo.tensor_search import add_docs
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from tests.marqo_test import MarqoTestCase
from marqo.s2_inference.s2_inference import Modality

def mock_image_preprocessor(image_path, return_tensors='pt'):
    return {'pixel_values': torch.rand(1, 3, 224, 224)}

def mock_video_preprocessor(video_path, return_tensors='pt'):
    return {'pixel_values': torch.rand(1, 3, 8, 224, 224)}

def mock_audio_preprocessor(audio_path, return_tensors='pt'):
    return {'pixel_values': torch.rand(1, 3, 112, 1036)} 

class MockMultimodalModel:
    def __init__(self, *args, **kwargs):
        pass

    def preprocessor(self, modality):
        if modality == Modality.IMAGE:
            return mock_image_preprocessor
        elif modality == Modality.VIDEO:
            return mock_video_preprocessor
        elif modality == Modality.AUDIO:
            return mock_audio_preprocessor
        else:
            return None

def mock_load_multimodal_model_and_get_preprocessors(*args, **kwargs):
    mock_model = MockMultimodalModel()
    mock_preprocessors = {
        "image": mock_image_preprocessor,
        "video": mock_model.preprocessor(Modality.VIDEO),
        "audio": mock_model.preprocessor(Modality.AUDIO),
        "text": None
    }
    return mock_model, mock_preprocessors

class TestAddDocumentsCombined(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        structured_image_index_request = cls.structured_marqo_index_request(
            name="structured_image_index" + str(uuid.uuid4()).replace('-', ''),
            fields=[
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer),
                FieldRequest(name="text_field_1", type=FieldType.Text,
                             features=[FieldFeature.Filter, FieldFeature.LexicalSearch])
            ],
            model=Model(name="open_clip/ViT-B-32/laion2b_s34b_b79k"),
            tensor_fields=["image_field_1", "text_field_1"]
        )

        structured_languagebind_index_request = cls.structured_marqo_index_request(
            name="my-multimodal-index" + str(uuid.uuid4()).replace('-', ''),
            fields=[
                FieldRequest(name="text_field_1", type=FieldType.Text),
                FieldRequest(name="text_field_2", type=FieldType.Text),
                FieldRequest(name="text_field_3", type=FieldType.Text),
                FieldRequest(name="video_field_1", type=FieldType.VideoPointer),
                FieldRequest(name="video_field_2", type=FieldType.VideoPointer),
                FieldRequest(name="video_field_3", type=FieldType.VideoPointer),
                FieldRequest(name="audio_field_1", type=FieldType.AudioPointer),
                FieldRequest(name="audio_field_2", type=FieldType.AudioPointer),
                FieldRequest(name="image_field_1", type=FieldType.ImagePointer),
                FieldRequest(name="image_field_2", type=FieldType.ImagePointer),
                FieldRequest(
                    name="multimodal_field",
                    type=FieldType.MultimodalCombination,
                    dependent_fields={
                        "text_field_1": 0.1,
                        "text_field_2": 0.1,
                        "image_field_1": 0.5,
                        "video_field_1": 0.1,
                        "video_field_2": 0.1,
                        "audio_field_1": 0.1
                    }
                )
            ],
            model=Model(name="LanguageBind"),
            tensor_fields=["multimodal_field", "text_field_3",
                        "video_field_3", "audio_field_2", "image_field_2"],
            normalize_embeddings=True,
        )

        unstructured_image_index_request = cls.unstructured_marqo_index_request(
            name="unstructured_image_index" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="open_clip/ViT-B-32/laion2b_s34b_b79k"),
            treat_urls_and_pointers_as_images=True
        )

        unstructured_languagebind_index_request = cls.unstructured_marqo_index_request(
            name="unstructured_languagebind_index" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="LanguageBind"),
            treat_urls_and_pointers_as_images=True,
            treat_urls_and_pointers_as_media=True
        )

        cls.indexes = cls.create_indexes([
            structured_image_index_request,
            structured_languagebind_index_request,
            unstructured_image_index_request,
            unstructured_languagebind_index_request
        ])

        cls.structured_marqo_index_name = structured_image_index_request.name
        cls.structured_languagebind_index_name = structured_languagebind_index_request.name
        cls.unstructured_marqo_index_name = unstructured_image_index_request.name
        cls.unstructured_languagebind_index_name = unstructured_languagebind_index_request.name

    def setUp(self) -> None:
        super().setUp()

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def test_add_documents_with_truncated_image(self):
        """Test to ensure that the add_documents API can properly return 400 for the document with a truncated image."""
        truncated_image_url = "https://marqo-assets.s3.amazonaws.com/tests/images/truncated_image.jpg"

        documents = [
            {
                "image_field_1": "https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_statue.png",
                "text_field_1": "This is a valid image",
                "_id": "1"
            },
            {
                "image_field_1": truncated_image_url,
                "text_field_1": "This is a truncated image",
                "_id": "2"
            }
        ]

        for index_name in [self.structured_marqo_index_name, self.unstructured_marqo_index_name]:
            tensor_fields = ["image_field_1", "text_field_1"] if index_name == self.unstructured_marqo_index_name \
                else None
            with self.subTest(f"test add documents with truncated image for {index_name}"):
                r = tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index_name,
                        docs=documents,
                        tensor_fields=tensor_fields,)
                ).dict(exclude_none=True, by_alias=True)
                print(f"response: {r}")
                self.assertEqual(True, r["errors"])
                self.assertEqual(2, len(r["items"]))
                self.assertEqual(200, r["items"][0]["status"])
                self.assertEqual(400, r["items"][1]["status"])
                self.assertIn("image file is truncated", r["items"][1]["error"])

    def test_add_document_callVectoriseWithoutPassingEnableCache(self):
        """Ensure vectorise does not receive enable_cache when calling add_documents."""
        documents = [
            {
                "text_field_1": "Test test",
                "_id": "1"
            }
        ]
        dummy_return = [[1.0, ] * 512, ]
        for index_name in [self.structured_marqo_index_name, self.unstructured_marqo_index_name]:
            tensor_fields = ["text_field_1"] if index_name == self.unstructured_marqo_index_name \
                else None
            with self.subTest(index_name):
                with patch("marqo.s2_inference.s2_inference.vectorise", return_value=dummy_return) as mock_vectorise:
                    r = tensor_search.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index_name,
                            docs=documents,
                            tensor_fields=tensor_fields)
                    ).dict(exclude_none=True, by_alias=True)
                    mock_vectorise.assert_called_once()
                    args, kwargs = mock_vectorise.call_args
                    self.assertFalse("enable_cache" in kwargs, "enable_cache should not be passed to "
                                                               "vectorise for add_documents")
                mock_vectorise.reset_mock()

    def test_add_multimodal_single_documents(self):
        """ """
        documents = [
            {"video_field_3": "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/---QUuC4vJs_000084_000094.mp4", "_id": "1"},
            # Replace the audio link with something marqo-hosted
            {"audio_field_2": "https://audio-previews.elements.envatousercontent.com/files/187680354/preview.mp3", "_id": "2"}, 
            {"image_field_2": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png", "_id": "3"},
            {"text_field_3": "hello there padawan. Today you will begin your training to be a Jedi", "_id": "4"},
        ]
        dummy_return = [[1.0, ] * 768]
        for index_name in [self.structured_languagebind_index_name, self.unstructured_languagebind_index_name]:
            with self.subTest(index_name):
                with patch("marqo.s2_inference.s2_inference.load_multimodal_model_and_get_preprocessors", 
                           side_effect=mock_load_multimodal_model_and_get_preprocessors) as mock_load, \
                     patch("marqo.s2_inference.s2_inference.vectorise", return_value=dummy_return):
                    res = tensor_search.add_documents(
                        self.config, 
                        add_docs_params=AddDocsParams(
                            docs=documents,
                            index_name=index_name,
                            tensor_fields=["text_field_3", "image_field_2", "video_field_3", "audio_field_2"] if "unstructured" in index_name else None
                        )
                    )
                    for item in res.dict(exclude_none=True, by_alias=True)['items']:
                        self.assertEqual(200, item['status'])
                    
                    self.assertEqual(4, res.dict(exclude_none=True, by_alias=True)['_batch_response_stats']['success_count'])
                    self.assertEqual(0, res.dict(exclude_none=True, by_alias=True)['_batch_response_stats']['error_count'])
                    self.assertEqual(0, res.dict(exclude_none=True, by_alias=True)['_batch_response_stats']['failure_count'])

                    print(res)

                    #_check_get_docs(self)

                    def _check_get_docs(self):
                        get_res = tensor_search.get_documents_by_ids(
                            config=self.config, index_name=index_name,
                            document_ids=["1","2","3","4"],
                            show_vectors=True
                        ).dict(exclude_none=True, by_alias=True)

                        # Iterate through each document in the response. For each document, check that the length  of the tensor facets 
                        # print(len(doc["results"][0]["_tensor_facets"])) is 1 for 0, 3, and 4. But len is 25 for 2
                        # Then, for each tensor facet, check that the length of the embedding is 768
                        # Then, for each embedding, check that it is not equal to any other embedding

                        for i, doc in enumerate(get_res['results']):
                            tensor_facets = doc['_tensor_facets']
                            
                            # Check the length of tensor facets
                            if i in [0, 3, 4]:
                                self.assertEqual(len(tensor_facets), 1, f"Document {i} should have 1 tensor facet")
                            elif i == 2:
                                self.assertEqual(len(tensor_facets), 25, f"Document 2 should have 25 tensor facets")
                            
                            # Check embedding length and uniqueness
                            embeddings = []
                            for facet in tensor_facets:
                                embedding = facet['_embedding']
                                self.assertEqual(len(embedding), 768, f"Embedding length should be 768 for document {i}")
                                self.assertNotIn(embedding, embeddings, f"Duplicate embedding found in document {i}")
                                embeddings.append(embedding)

    def test_add_multimodal_field_document(self):
        multimodal_document = {
            "_id": "1_multimodal",
            "text_field_1": "New York",
            "text_field_2": "Los Angeles",
            "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
            "video_field_1": "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/---QUuC4vJs_000084_000094.mp4",
            "video_field_2": "https://marqo-k400-video-test-dataset.s3.amazonaws.com/videos/---QUuC4vJs_000084_000094.mp4",
            "audio_field_1": "https://audio-previews.elements.envatousercontent.com/files/187680354/preview.mp3",
        },
        for index_name in [self.structured_languagebind_index_name, self.unstructured_languagebind_index_name]:
            mappings = {
                "multimodal_field": {
                    "type": "multimodal_combination",
                    "weights": {
                        "text_field_1": 0.1, 
                        "text_field_2": 0.1, 
                        "image_field": 0.5, 
                        "video_field_1": 0.1, 
                        "video_field_2": 0.1, 
                        "audio_field_1": 0.1
                    },
                }
            } if "unstructured" in index_name else None
            res = tensor_search.add_documents(
                self.config, 
                add_docs_params=AddDocsParams(
                    docs=multimodal_document,
                    index_name=index_name,
                    tensor_fields=["multimodal_field"] if "unstructured" in index_name else None,
                    mappings=mappings
                )
            )

            print(res)
            for item in res.dict(exclude_none=True, by_alias=True)['items']:
                self.assertEqual(200, item['status'])


    def test_imageRepoHandleThreadHandleError_successfully(self):
        """Ensure content_repo can catch an unexpected error right in thread."""
        documents = [
            {
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/"
                                 "marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
                "_id": "1"
            }
        ]

        for index_name in [self.unstructured_marqo_index_name, self.structured_marqo_index_name]:
            error = Exception("Unexpected error during image download")
            tensor_fields = ["image_field_1"] if index_name == self.unstructured_marqo_index_name \
                else None
            with (self.subTest(f"{index_name}-{error}")):
                with patch("marqo.s2_inference.clip_utils.requests.get", side_effect=error) \
                        as mock_requests_get:
                    with self.assertRaises(Exception) as e:
                        r = tensor_search.add_documents(
                            config=self.config,
                            add_docs_params=AddDocsParams(
                                index_name=index_name,
                                docs=documents,
                                tensor_fields=tensor_fields)
                        ).dict(exclude_none=True, by_alias=True)
                        self.assertIn("Unexpected error during image download", str(e.exception))

    def test_addDocumentsPassTensorToVectorise(self):
        """Ensure vectorise receives tensor from add_documents when the model is OpenCLIP or CLIP."""
        documents = [
            {
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/"
                                 "marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
                "_id": "1"
            }
        ]
        dummy_return = [[1.0, ] * 512, ]
        for index_name in [self.structured_marqo_index_name, self.unstructured_marqo_index_name]:
            tensor_fields = ["image_field_1"] if index_name == self.unstructured_marqo_index_name \
                else None
            with self.subTest(index_name):
                with patch("marqo.s2_inference.s2_inference.vectorise", return_value=dummy_return) as mock_vectorise:
                    r = tensor_search.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(
                            index_name=index_name,
                            docs=documents,
                            tensor_fields=tensor_fields)
                    ).dict(exclude_none=True, by_alias=True)
                    mock_vectorise.assert_called_once()
                    args, kwargs = mock_vectorise.call_args
                    self.assertIn("content", kwargs)
                    content = kwargs["content"]
                    self.assertEqual(1, len(content))
                    self.assertEqual((3, 224, 224), content[0].shape)

    def test_downloadImagesThreadCount(self):
        """
        Test that image download thread count is respected
        """
        docs = [
            {"_id": str(i),
             "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/"
                            "assets/ai_hippo_realistic.png"
             } for i in range(10)
        ]
        for index_name in [self.structured_marqo_index_name, self.unstructured_marqo_index_name]:
            tensor_fields = ["image_field_1"] if index_name == self.unstructured_marqo_index_name \
                else None
            with self.subTest(index_name):
                for thread_count in [2, 5]:
                    with patch.object(
                            add_docs, 'threaded_download_and_preprocess_content',
                            wraps=add_docs.threaded_download_and_preprocess_content
                    ) as mock_download_images:
                        tensor_search.add_documents(
                            config=self.config, add_docs_params=AddDocsParams(
                                index_name=index_name, docs=docs, device="cpu",
                                image_download_thread_count=thread_count,
                                tensor_fields=tensor_fields
                            )
                        ).dict(exclude_none=True, by_alias=True)

                        self.assertEqual(thread_count, mock_download_images.call_count)

    def test_imageDownloadWithoutPreprocessor(self):
        content_repo = dict()
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        test_doc = {
            'field_1': 'https://google.com/my_dog.png',  # error because such an image doesn't exist
            'field_2': good_url
        }

        add_docs.threaded_download_and_preprocess_content(
            allocated_docs=[test_doc],
            content_repo=content_repo,
            tensor_fields=['field_1', 'field_2'],
            image_download_headers={},
        )
        assert len(content_repo) == 2
        assert isinstance(content_repo['https://google.com/my_dog.png'], PIL.UnidentifiedImageError)
        assert isinstance(content_repo[good_url], types.ImageType)

    def test_imageDownloadWithPreprocessor(self):
        content_repo = dict()
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        test_doc = {
            'field_1': 'https://google.com/my_dog.png',  # error because such an image doesn't exist
            'field_2': good_url
        }

        add_docs.threaded_download_and_preprocess_content(
            allocated_docs=[test_doc],
            content_repo=content_repo,
            tensor_fields=['field_1', 'field_2'],
            image_download_headers={},
            preprocessors={'image': lambda x: torch.randn(3, 224, 224)},
            device='cpu'
        )
        assert len(content_repo) == 2
        assert isinstance(content_repo['https://google.com/my_dog.png'], PIL.UnidentifiedImageError)
        assert isinstance(content_repo[good_url], Tensor)

    def test_image_download_timeout(self):
        mock_get = mock.MagicMock()
        mock_get.side_effect = requests.exceptions.RequestException

        @mock.patch('requests.get', mock_get)
        def run():
            content_repo = dict()
            add_docs.threaded_download_and_preprocess_content(
                allocated_docs=[
                    {"Title": "frog", "Desc": "blah"}, {"Title": "Dog", "Loc": "https://google.com/my_dog.png"}],
                content_repo=content_repo,
                tensor_fields=['Title', 'Desc', 'Loc'],
                image_download_headers={}
            )
            assert list(content_repo.keys()) == ['https://google.com/my_dog.png']
            assert isinstance(content_repo['https://google.com/my_dog.png'], PIL.UnidentifiedImageError)
            return True

        assert run()

    def test_resilient_add_images(self):
        """
        Various image URLs are handled correctly
        """
        docs_results = [
            ([{"_id": "123",
               "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"},
              {"_id": "789",
               "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"},
              {"_id": "456", "image_field_1": "https://www.marqo.ai/this/image/doesnt/exist.png"}],
             [("123", 200), ("789", 200), ("456", 400)]
             ),
            ([{"_id": "123",
               "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png"},
              {"_id": "789",
               "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png"},
              {"_id": "456", "image_field_1": "https://www.marqo.ai/this/image/doesnt/exist.png"},
              {"_id": "111", "image_field_1": "https://www.marqo.ai/this/image/doesnt/exist2.png"}],
             [("123", 200), ("789", 200), ("456", 400), ("111", 400)]
             ),
            ([{"_id": "505", "image_field_1": "https://www.marqo.ai/this/image/doesnt/exist3.png"},
              {"_id": "456", "image_field_1": "https://www.marqo.ai/this/image/doesnt/exist.png"},
              {"_id": "111", "image_field_1": "https://www.marqo.ai/this/image/doesnt/exist2.png"}],
             [("505", 400), ("456", 400), ("111", 400)]
             ),
            ([{"_id": "505", "image_field_1": "https://www.marqo.ai/this/image/doesnt/exist2.png"}],
             [("505", 400)]
             ),
        ]
        for index_name in [self.structured_marqo_index_name, self.unstructured_marqo_index_name]:
            tensor_fields = ["image_field_1"] if index_name == self.unstructured_marqo_index_name \
                else None
            with self.subTest(index_name):
                for docs, expected_results in docs_results:
                    with self.subTest(f'{expected_results} - {index_name}'):
                        add_res = tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                            index_name=index_name, docs=docs, device="cpu", tensor_fields=tensor_fields)).dict(
                            exclude_none=True, by_alias=True)
                        self.assertEqual(len(expected_results), len(add_res['items']))
                        for i, res_dict in enumerate(add_res['items']):
                            self.assertEqual(expected_results[i][0], res_dict["_id"], res_dict)
                            self.assertEqual(expected_results[i][1], res_dict['status'], res_dict)

    def test_threaded_download_images_non_tensor_field(self):
        """Tests add_docs.threaded_download_images(). URLs not in tensor fields should not be downloaded """
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        bad_url = 'https://google.com/my_dog.png'
        examples = [
            ([{
                'field_1': bad_url,
                'field_2': good_url
            }], {
                 bad_url: PIL.UnidentifiedImageError,
                 good_url: types.ImageType
             }),
            ([{
                'nt_1': bad_url,
                'nt_2': good_url
            }], {}),
            ([{
                'field_1': bad_url,
                'nt_1': good_url
            }], {
                 bad_url: PIL.UnidentifiedImageError,
             }),
            ([{
                'nt_2': bad_url,
                'field_2': good_url
            }], {
                 good_url: types.ImageType
             }),
        ]
        for docs, expected_repo_structure in examples:
            content_repo = dict()
            add_docs.threaded_download_and_preprocess_content(
                allocated_docs=docs,
                content_repo=content_repo,
                tensor_fields=['field_1', 'field_2'],
                image_download_headers={}
            )
            assert len(expected_repo_structure) == len(content_repo)
            for k in expected_repo_structure:
                assert isinstance(content_repo[k], expected_repo_structure[k])

    def test_download_images_non_tensor_field(self):
        """tests add_docs.download_images(). URLs not in tensor fields should not be downloaded """
        good_url = 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png'
        bad_url = 'https://google.com/my_dog.png'
        examples = [
            ([{
                'field_1': bad_url,
                'field_2': good_url
            }], {
                 bad_url: PIL.UnidentifiedImageError,
                 good_url: Tensor
             }),
            ([{
                'nt_1': bad_url,
                'nt_2': good_url
            }], {}),
            ([{
                'field_1': bad_url,
                'nt_1': good_url
            }], {
                 bad_url: PIL.UnidentifiedImageError,
             }),
            ([{
                'nt_2': bad_url,
                'field_2': good_url
            }], {
                 good_url: Tensor
             }),
        ]
        model_properties = (
            {
                "name": "ViT-B/32",
                "dimensions": 512,
                "notes": "CLIP ViT-B/32",
                "type": "clip",
            }
        )

        for docs, expected_repo_structure in examples:
            with mock.patch('PIL.Image.Image.close') as mock_close:
                with add_docs.download_and_preprocess_content(
                    docs=docs,
                    thread_count=20,
                    tensor_fields=['field_1', 'field_2'],
                    image_download_headers={},
                    model_name="ViT-B/32",
                    normalize_embeddings=True,
                    model_properties=model_properties,
                    device="cpu"
                ) as content_repo:
                    self.assertEqual(len(expected_repo_structure), len(content_repo))
                    for k in expected_repo_structure:
                        print(f"expected_repo_structure[k] = {expected_repo_structure[k]}")
                        print(f"content_repo[k] = {content_repo[k]}")
                        self.assertIsInstance(content_repo[k], expected_repo_structure[k])

            # Images should not be closed as they are Tensor instead of ImageType
            mock_close.assert_not_called()

    def test_idErrorWhenImageDownloading(self):
        """A test ensure image download is not raising 500 error when there is an invalid _id.

        Image download use the document _id to generate a unique thread id.
        However, the image download happens before validate the document _id.
        This test ensures that the image download does not raise a 500 error when the document _id is invalid.
        """
        test_docs = [
            {
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline"
                                 "/examples/ImageSearchGuide/data/image1.jpg",
                 "text_field_1": "this is a valid image",
                 "_id": "1"
            },
            {
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline"
                                 "/examples/ImageSearchGuide/data/image2.jpg",
                "text_field_1": "this is a invalid image due to int id",
                "_id": 2
            },
            {
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline"
                                 "/examples/ImageSearchGuide/data/image3.jpg",
                "text_field_1": "this is a invalid image due to None",
                "_id": None
            },
            {
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline"
                                 "/examples/ImageSearchGuide/data/image4.jpg",
                "text_field_1": "this is a invalid image due to ",
                "_id": []
            }
        ]

        for index_name in [self.unstructured_marqo_index_name, self.structured_marqo_index_name]:
            tensor_fields = ["image_field_1", "text_field_1"] if index_name == self.unstructured_marqo_index_name \
                else None
            with self.subTest(index_name):
                r = tensor_search.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index_name,
                        docs=test_docs,
                        tensor_fields=tensor_fields)
                ).dict(exclude_none=True, by_alias=True)
                self.assertEqual(True, r["errors"])
                self.assertEqual(4, len(r["items"]))
                self.assertEqual(200, r["items"][0]["status"])
                for i in range(1, 4):
                    self.assertEqual(400, r["items"][i]["status"])
                    self.assertIn("Document _id must be a string", r["items"][i]["error"])