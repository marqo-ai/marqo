import unittest
from unittest.mock import Mock, patch, MagicMock
from PIL import UnidentifiedImageError
import torch
from marqo.s2_inference.errors import UnsupportedModalityError, S2InferenceError
from marqo.tensor_search.add_docs import threaded_download_and_preprocess_content
from marqo.core.models.marqo_index import IndexType, MarqoIndex, FieldType
from marqo.s2_inference.s2_inference import Modality
from marqo.s2_inference.models.model_type import ModelType
from marqo.tensor_search.telemetry import RequestMetricsStore, RequestMetrics


class TestThreadedDownloadAndPreprocess(unittest.TestCase):

    def setUp(self):
        self.mock_image_url = "https://example.com/image.jpg"
        self.mock_video_url = "https://example.com/video.mp4"
        self.mock_audio_url = "https://example.com/audio.mp3"
        self.mock_text = "This is some text content"

        self.mock_image = Mock()
        self.mock_image_tensor = torch.tensor([1, 2, 3])
        self.mock_video_chunks = [{"tensor": torch.tensor([4, 5, 6])}]
        self.mock_audio_chunks = [{"tensor": torch.tensor([7, 8, 9])}]

        # Create a mock for the model
        self.mock_model = Mock()
        self.mock_model.properties = {
            "type": ModelType.CLIP,
            "supported_modalities": [Modality.IMAGE, Modality.TEXT]
        }

        # Create the MarqoIndex mock and set its attributes
        self.mock_marqo_index = Mock(spec=MarqoIndex)
        self.mock_marqo_index.type = IndexType.Unstructured
        self.mock_marqo_index.model = self.mock_model

        # Mock RequestMetricsStore
        self.mock_metrics = Mock(spec=RequestMetrics)
        
        # Create a mock context manager for the time method
        mock_time_context = MagicMock()
        mock_time_context.__enter__ = MagicMock()
        mock_time_context.__exit__ = MagicMock()
        self.mock_metrics.time.return_value = mock_time_context

        patcher = patch('marqo.tensor_search.add_docs.RequestMetricsStore', autospec=True)
        self.mock_metrics_store = patcher.start()
        self.mock_metrics_store.for_request.return_value = self.mock_metrics
        self.addCleanup(patcher.stop)

    @patch("marqo.tensor_search.add_docs.clip_utils.load_image_from_path")
    @patch("marqo.tensor_search.add_docs.infer_modality")
    def test_image_unstructured_index(self, mock_infer_modality, mock_load_image):
        mock_infer_modality.return_value = Modality.IMAGE
        mock_load_image.return_value = self.mock_image

        docs = [{"field1": self.mock_image_url}]
        media_repo = {}
        tensor_fields = ["field1"]
        
        threaded_download_and_preprocess_content(
            docs, media_repo, tensor_fields, {}, device="cpu",
            marqo_index=self.mock_marqo_index
        )

        self.assertIn(self.mock_image_url, media_repo)
        self.assertEqual(media_repo[self.mock_image_url], self.mock_image)

    @patch("marqo.tensor_search.add_docs.clip_utils.load_image_from_path")
    @patch("marqo.tensor_search.add_docs.infer_modality")
    def test_image_structured_index(self, mock_infer_modality, mock_load_image):
        mock_infer_modality.return_value = Modality.IMAGE
        mock_load_image.return_value = self.mock_image

        self.mock_marqo_index.type = IndexType.Structured
        docs = [{"field1": self.mock_image_url}]
        media_repo = {}
        tensor_fields = ["field1"]
        media_field_types_mapping = {"field1": FieldType.ImagePointer}
        
        threaded_download_and_preprocess_content(
            docs, media_repo, tensor_fields, {}, device="cpu",
            marqo_index=self.mock_marqo_index,
            media_field_types_mapping=media_field_types_mapping
        )

        self.assertIn(self.mock_image_url, media_repo)
        self.assertEqual(media_repo[self.mock_image_url], self.mock_image)

    @patch("marqo.tensor_search.add_docs.download_and_chunk_media")
    @patch("marqo.tensor_search.add_docs.infer_modality")
    def test_video_unstructured_index(self, mock_infer_modality, mock_download_and_chunk):
        mock_infer_modality.return_value = Modality.VIDEO
        mock_download_and_chunk.return_value = self.mock_video_chunks

        self.mock_model.properties["type"] = ModelType.LanguageBind
        self.mock_model.properties["supported_modalities"].append(Modality.VIDEO)

        docs = [{"field1": self.mock_video_url}]
        media_repo = {}
        tensor_fields = ["field1"]
        
        threaded_download_and_preprocess_content(
            docs, media_repo, tensor_fields, {}, device="cpu",
            marqo_index=self.mock_marqo_index
        )

        self.assertIn(self.mock_video_url, media_repo)
        self.assertEqual(media_repo[self.mock_video_url], self.mock_video_chunks)

    @patch("marqo.tensor_search.add_docs.download_and_chunk_media")
    @patch("marqo.tensor_search.add_docs.infer_modality")
    def test_audio_structured_index(self, mock_infer_modality, mock_download_and_chunk):
        mock_infer_modality.return_value = Modality.AUDIO
        mock_download_and_chunk.return_value = self.mock_audio_chunks

        self.mock_marqo_index.type = IndexType.Structured
        self.mock_model.properties["type"] = ModelType.LanguageBind
        self.mock_model.properties["supported_modalities"].append(Modality.AUDIO)

        docs = [{"field1": self.mock_audio_url}]
        media_repo = {}
        tensor_fields = ["field1"]
        media_field_types_mapping = {"field1": FieldType.AudioPointer}
        
        threaded_download_and_preprocess_content(
            docs, media_repo, tensor_fields, {}, device="cpu",
            marqo_index=self.mock_marqo_index,
            media_field_types_mapping=media_field_types_mapping
        )

        self.assertIn(self.mock_audio_url, media_repo)
        self.assertEqual(media_repo[self.mock_audio_url], self.mock_audio_chunks)

    @patch("marqo.tensor_search.add_docs.infer_modality")
    def test_unsupported_modality(self, mock_infer_modality):
        mock_infer_modality.return_value = Modality.VIDEO

        docs = [{"field1": self.mock_video_url}]
        media_repo = {}
        tensor_fields = ["field1"]
        
        threaded_download_and_preprocess_content(
            docs, media_repo, tensor_fields, {}, device="cpu",
            marqo_index=self.mock_marqo_index
        )

        self.assertIn(self.mock_video_url, media_repo)
        self.assertIsInstance(media_repo[self.mock_video_url], UnsupportedModalityError)

    @patch("marqo.tensor_search.add_docs.clip_utils.load_image_from_path")
    @patch("marqo.tensor_search.add_docs.infer_modality")
    def test_image_load_error(self, mock_infer_modality, mock_load_image):
        mock_infer_modality.return_value = Modality.IMAGE
        mock_load_image.side_effect = UnidentifiedImageError()

        docs = [{"field1": self.mock_image_url}]
        media_repo = {}
        tensor_fields = ["field1"]
        
        threaded_download_and_preprocess_content(
            docs, media_repo, tensor_fields, {}, device="cpu",
            marqo_index=self.mock_marqo_index
        )

        self.assertIn(self.mock_image_url, media_repo)
        self.assertIsInstance(media_repo[self.mock_image_url], UnidentifiedImageError)

    @patch("marqo.tensor_search.add_docs.download_and_chunk_media")
    @patch("marqo.tensor_search.add_docs.infer_modality")
    def test_video_processing_error(self, mock_infer_modality, mock_download_and_chunk):
        mock_infer_modality.return_value = Modality.VIDEO
        mock_download_and_chunk.side_effect = S2InferenceError("Processing error")

        self.mock_model.properties["type"] = ModelType.LanguageBind
        self.mock_model.properties["supported_modalities"].append(Modality.VIDEO)

        docs = [{"field1": self.mock_video_url}]
        media_repo = {}
        tensor_fields = ["field1"]
        
        threaded_download_and_preprocess_content(
            docs, media_repo, tensor_fields, {}, device="cpu",
            marqo_index=self.mock_marqo_index
        )

        self.assertIn(self.mock_video_url, media_repo)
        self.assertIsInstance(media_repo[self.mock_video_url], S2InferenceError)

    @patch("marqo.tensor_search.add_docs.download_and_chunk_media")
    @patch("marqo.tensor_search.add_docs.infer_modality")
    def test_video_and_audio_unstructured_index(self, mock_infer_modality, mock_download_and_chunk):
        # Set up the mock model to support both video and audio
        self.mock_model.properties["type"] = ModelType.LanguageBind
        self.mock_model.properties["supported_modalities"] = [Modality.VIDEO, Modality.AUDIO, Modality.TEXT]

        # Test data
        docs = [
            {"field1": self.mock_video_url},
            {"field2": self.mock_audio_url}
        ]
        media_repo = {}
        tensor_fields = ["field1", "field2"]

        # Mock the infer_modality and download_and_chunk_media functions
        mock_infer_modality.side_effect = [Modality.VIDEO, Modality.AUDIO]
        mock_download_and_chunk.side_effect = [self.mock_video_chunks, self.mock_audio_chunks]

        # Call the function
        threaded_download_and_preprocess_content(
            docs, media_repo, tensor_fields, {}, device="cpu",
            marqo_index=self.mock_marqo_index
        )

        # Assertions
        self.assertIn(self.mock_video_url, media_repo)
        self.assertIn(self.mock_audio_url, media_repo)
        self.assertEqual(media_repo[self.mock_video_url], self.mock_video_chunks)
        self.assertEqual(media_repo[self.mock_audio_url], self.mock_audio_chunks)

        # Verify that download_and_chunk_media was called twice
        self.assertEqual(mock_download_and_chunk.call_count, 2)

        # Verify the calls to download_and_chunk_media
        mock_download_and_chunk.assert_any_call(
            self.mock_video_url, "cpu", None, Modality.VIDEO, self.mock_marqo_index, None
        )
        mock_download_and_chunk.assert_any_call(
            self.mock_audio_url, "cpu", None, Modality.AUDIO, self.mock_marqo_index, None
        )