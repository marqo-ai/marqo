from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, Mock, mock_open, patch

import botocore.exceptions
from model_management.services.errors import ModelDownloadFailedError
from model_management.services.model_manager.triton_model_downloader import (
    TritonModelDownloader,
)


class TestTritonModelDownloader(TestCase):
    """Test class for TritonModelDownloader."""

    def setUp(self):
        """Set up test fixtures."""
        self.sources = ["s3://bucket/model.onnx"]
        self.base_dir = "/tmp/models"
        self.model_name = "test-model"
        self.config_pbtxt = 'name: "test-model"\nmax_batch_size: 8'

    def test_triton_model_downloader_initialization(self):
        """Test TritonModelDownloader initialization."""
        downloader = TritonModelDownloader(
            sources=self.sources,
            base_dir=self.base_dir,
            model_name=self.model_name,
            config_pbtxt=self.config_pbtxt,
            overwrite=False,
        )

        self.assertEqual(self.sources, downloader.sources)
        self.assertEqual(Path(self.base_dir), downloader.base_dir)
        self.assertEqual(self.model_name, downloader.model_name)
        self.assertEqual(self.config_pbtxt, downloader.config_pbtxt)
        self.assertFalse(downloader.overwrite)

    def test_triton_model_downloader_initialization_with_overwrite(self):
        """Test TritonModelDownloader initialization with overwrite=True."""
        downloader = TritonModelDownloader(
            sources=self.sources,
            base_dir=self.base_dir,
            model_name=self.model_name,
            overwrite=True,
        )

        self.assertTrue(downloader.overwrite)

    def test_triton_model_downloader_initialization_without_config(self):
        """Test TritonModelDownloader initialization without config_pbtxt."""
        downloader = TritonModelDownloader(
            sources=self.sources, base_dir=self.base_dir, model_name=self.model_name
        )

        self.assertIsNone(downloader.config_pbtxt)

    @patch("model_management.services.model_manager.triton_model_downloader.Path.mkdir")
    @patch(
        "model_management.services.model_manager.triton_model_downloader.Path.write_text"
    )
    def test_version_dir_creates_directory_structure(self, mock_write_text, mock_mkdir):
        """Test _version_dir creates correct directory structure."""
        downloader = TritonModelDownloader(
            sources=self.sources,
            base_dir=self.base_dir,
            model_name=self.model_name,
            config_pbtxt=self.config_pbtxt,
        )

        version_dir = downloader._version_dir()

        # Verify directory was created
        mock_mkdir.assert_called()

        # Verify config.pbtxt was written
        mock_write_text.assert_called_once_with(self.config_pbtxt)

        # Verify returned path is correct
        self.assertIn("test-model", str(version_dir))
        self.assertIn("1", str(version_dir))

    @patch("model_management.services.model_manager.triton_model_downloader.Path.mkdir")
    @patch(
        "model_management.services.model_manager.triton_model_downloader.Path.write_text"
    )
    def test_version_dir_without_config_pbtxt(self, mock_write_text, mock_mkdir):
        """Test _version_dir without config_pbtxt."""
        downloader = TritonModelDownloader(
            sources=self.sources,
            base_dir=self.base_dir,
            model_name=self.model_name,
            config_pbtxt=None,
        )

        _ = downloader._version_dir()

        # Verify directory was created
        mock_mkdir.assert_called()

        # Verify config.pbtxt was NOT written
        mock_write_text.assert_not_called()

    @patch("builtins.open", new_callable=mock_open)
    @patch("model_management.services.model_manager.triton_model_downloader.tqdm")
    def test_download_with_progress_success(self, mock_tqdm, mock_file):
        """Test _download_with_progress downloads file successfully."""
        mock_fs = Mock()
        mock_fs.info = Mock(return_value={"size": 1024})
        mock_fs.open = Mock(return_value=MagicMock())

        # Mock file read to return data in chunks
        mock_fs.open.return_value.__enter__.return_value.read = Mock(
            side_effect=[b"chunk1", b"chunk2", b""]
        )

        downloader = TritonModelDownloader(
            sources=self.sources, base_dir=self.base_dir, model_name=self.model_name
        )

        downloader._download_with_progress(
            mock_fs, "s3://bucket/model.onnx", Path("/tmp/model.onnx")
        )

        # Verify fs.info was called
        mock_fs.info.assert_called_once_with("s3://bucket/model.onnx")

        # Verify fs.open was called
        mock_fs.open.assert_called_once()

    def test_download_with_progress_no_credentials_error(self):
        """Test _download_with_progress raises ModelDownloadFailedError on NoCredentialsError."""
        mock_fs = Mock()
        mock_fs.info = Mock(side_effect=botocore.exceptions.NoCredentialsError())

        downloader = TritonModelDownloader(
            sources=self.sources, base_dir=self.base_dir, model_name=self.model_name
        )

        with self.assertRaises(ModelDownloadFailedError) as context:
            downloader._download_with_progress(
                mock_fs, "s3://bucket/model.onnx", Path("/tmp/model.onnx")
            )

        self.assertIn("AWS credentials", str(context.exception))

    def test_download_with_progress_file_not_found_error(self):
        """Test _download_with_progress raises ModelDownloadFailedError on FileNotFoundError."""
        mock_fs = Mock()
        mock_fs.info = Mock(side_effect=FileNotFoundError("File not found"))

        downloader = TritonModelDownloader(
            sources=self.sources, base_dir=self.base_dir, model_name=self.model_name
        )

        with self.assertRaises(ModelDownloadFailedError) as context:
            downloader._download_with_progress(
                mock_fs, "s3://bucket/model.onnx", Path("/tmp/model.onnx")
            )

        self.assertIn("not found", str(context.exception))

    @patch(
        "model_management.services.model_manager.triton_model_downloader.Path.exists"
    )
    @patch("model_management.services.model_manager.triton_model_downloader.Path.mkdir")
    @patch(
        "model_management.services.model_manager.triton_model_downloader.fsspec.core.url_to_fs"
    )
    @patch.object(TritonModelDownloader, "_download_with_progress")
    @patch.object(TritonModelDownloader, "_version_dir")
    def test_prepare_and_download_single_source(
        self, mock_version_dir, mock_download, mock_url_to_fs, mock_mkdir, mock_exists
    ):
        """Test prepare_and_download with single source."""
        mock_version_dir.return_value = Path("/tmp/models/test-model/1")
        mock_exists.return_value = False
        mock_fs = Mock()
        mock_url_to_fs.return_value = (mock_fs, "s3://bucket/model.onnx")

        downloader = TritonModelDownloader(
            sources=["s3://bucket/model.onnx"],
            base_dir=self.base_dir,
            model_name=self.model_name,
        )

        out_paths = downloader.prepare_and_download()

        # Verify download was called
        mock_download.assert_called_once()

        # Verify output paths
        self.assertEqual(1, len(out_paths))
        self.assertIn("model.onnx", str(out_paths[0]))

    @patch(
        "model_management.services.model_manager.triton_model_downloader.Path.exists"
    )
    @patch("model_management.services.model_manager.triton_model_downloader.Path.mkdir")
    @patch(
        "model_management.services.model_manager.triton_model_downloader.fsspec.core.url_to_fs"
    )
    @patch.object(TritonModelDownloader, "_download_with_progress")
    @patch.object(TritonModelDownloader, "_version_dir")
    def test_prepare_and_download_multiple_sources(
        self, mock_version_dir, mock_download, mock_url_to_fs, mock_mkdir, mock_exists
    ):
        """Test prepare_and_download with multiple sources."""
        mock_version_dir.return_value = Path("/tmp/models/test-model/1")
        mock_exists.return_value = False
        mock_fs = Mock()
        mock_url_to_fs.return_value = (mock_fs, "dummy_path")

        downloader = TritonModelDownloader(
            sources=["s3://bucket/model.onnx", "s3://bucket/model.onnx.data"],
            base_dir=self.base_dir,
            model_name=self.model_name,
        )

        out_paths = downloader.prepare_and_download()

        # Verify download was called twice
        self.assertEqual(2, mock_download.call_count)

        # Verify output paths
        self.assertEqual(2, len(out_paths))

    @patch(
        "model_management.services.model_manager.triton_model_downloader.Path.exists"
    )
    @patch.object(TritonModelDownloader, "_download_with_progress")
    @patch.object(TritonModelDownloader, "_version_dir")
    def test_prepare_and_download_skips_existing_files(
        self, mock_version_dir, mock_download, mock_exists
    ):
        """Test prepare_and_download skips existing files when overwrite=False."""
        mock_version_dir.return_value = Path("/tmp/models/test-model/1")
        mock_exists.return_value = True  # File already exists

        downloader = TritonModelDownloader(
            sources=["s3://bucket/model.onnx"],
            base_dir=self.base_dir,
            model_name=self.model_name,
            overwrite=False,
        )

        out_paths = downloader.prepare_and_download()

        # Verify download was NOT called (file already exists)
        mock_download.assert_not_called()

        # Verify output paths still returned
        self.assertEqual(1, len(out_paths))

    @patch(
        "model_management.services.model_manager.triton_model_downloader.Path.exists"
    )
    @patch("model_management.services.model_manager.triton_model_downloader.Path.mkdir")
    @patch(
        "model_management.services.model_manager.triton_model_downloader.fsspec.core.url_to_fs"
    )
    @patch.object(TritonModelDownloader, "_download_with_progress")
    @patch.object(TritonModelDownloader, "_version_dir")
    def test_prepare_and_download_overwrites_existing_files(
        self, mock_version_dir, mock_download, mock_url_to_fs, mock_mkdir, mock_exists
    ):
        """Test prepare_and_download overwrites existing files when overwrite=True."""
        mock_version_dir.return_value = Path("/tmp/models/test-model/1")
        mock_exists.return_value = True  # File already exists
        mock_fs = Mock()
        mock_url_to_fs.return_value = (mock_fs, "s3://bucket/model.onnx")

        downloader = TritonModelDownloader(
            sources=["s3://bucket/model.onnx"],
            base_dir=self.base_dir,
            model_name=self.model_name,
            overwrite=True,
        )

        out_paths = downloader.prepare_and_download()

        # Verify download WAS called (overwrite=True)
        mock_download.assert_called_once()

        # Verify output paths returned
        self.assertEqual(1, len(out_paths))

    @patch(
        "model_management.services.model_manager.triton_model_downloader.Path.exists"
    )
    @patch("model_management.services.model_manager.triton_model_downloader.Path.mkdir")
    @patch(
        "model_management.services.model_manager.triton_model_downloader.fsspec.core.url_to_fs"
    )
    @patch.object(TritonModelDownloader, "_download_with_progress")
    @patch.object(TritonModelDownloader, "_version_dir")
    def test_prepare_and_download_with_various_sources(
        self, mock_version_dir, mock_download, mock_url_to_fs, mock_mkdir, mock_exists
    ):
        """Test prepare_and_download with various source formats."""
        test_cases = [
            ["s3://bucket/model.onnx"],
            ["http://example.com/model.onnx"],
            ["https://cdn.example.com/models/model.onnx"],
            ["/local/path/model.onnx"],
        ]

        mock_version_dir.return_value = Path("/tmp/models/test-model/1")
        mock_exists.return_value = False
        mock_fs = Mock()
        mock_url_to_fs.return_value = (mock_fs, "dummy_path")

        for sources in test_cases:
            with self.subTest(sources=sources):
                mock_download.reset_mock()
                downloader = TritonModelDownloader(
                    sources=sources, base_dir=self.base_dir, model_name=self.model_name
                )

                out_paths = downloader.prepare_and_download()

                self.assertEqual(1, len(out_paths))
                mock_download.assert_called_once()
