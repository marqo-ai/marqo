"""Integration tests for TritonModelDownloader with real model downloads."""

import os
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

from model_management.services.model_manager.triton_model_downloader import (
    TritonModelDownloader,
)

# Real model properties from actual Marqo models
test_model_sources = {
    "all-MiniLM-L6-v2-text-encoder": [
        "s3://marqo-opensource-models/sentence-transformers-all-minilm-l6-v2/model.onnx"
    ]
}


class TestTritonModelDownloader(TestCase):
    """Integration tests for TritonModelDownloader with basic functionality."""

    def setUp(self):
        """Set up test fixtures for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_downloader_initialization(self):
        """Test that TritonModelDownloader initializes with correct parameters."""
        sources = ["s3://bucket/model.onnx"]
        model_name = "test-model"
        config_pbtxt = "name: test-model"

        downloader = TritonModelDownloader(
            sources=sources,
            base_dir=self.temp_dir,
            model_name=model_name,
            config_pbtxt=config_pbtxt,
            overwrite=False,
        )

        self.assertEqual(sources, downloader.sources)
        self.assertEqual(Path(self.temp_dir), downloader.base_dir)
        self.assertEqual(model_name, downloader.model_name)
        self.assertEqual(config_pbtxt, downloader.config_pbtxt)
        self.assertFalse(downloader.overwrite)

    def test_downloader_initialization_without_config(self):
        """Test initialization without config.pbtxt."""
        downloader = TritonModelDownloader(
            sources=["s3://bucket/model.onnx"],
            base_dir=self.temp_dir,
            model_name="test-model",
            config_pbtxt=None,
            overwrite=True,
        )

        self.assertIsNone(downloader.config_pbtxt)
        self.assertTrue(downloader.overwrite)

    def test_version_dir_creates_config_pbtxt(self):
        """Test that _version_dir creates config.pbtxt when provided."""
        model_name = "test-model"
        config_content = "name: test-model\nmax_batch_size: 8"

        downloader = TritonModelDownloader(
            sources=[],
            base_dir=self.temp_dir,
            model_name=model_name,
            config_pbtxt=config_content,
            overwrite=False,
        )

        downloader._version_dir()

        # Verify config.pbtxt was created with correct content
        config_path = os.path.join(self.temp_dir, model_name, "config.pbtxt")
        self.assertTrue(os.path.exists(config_path))

        with open(config_path, "r") as f:
            actual_content = f.read()
        self.assertEqual(config_content, actual_content)

    def test_version_dir_does_not_create_config_when_none(self):
        """Test that _version_dir does not create config.pbtxt when None."""
        model_name = "test-model"

        downloader = TritonModelDownloader(
            sources=[],
            base_dir=self.temp_dir,
            model_name=model_name,
            config_pbtxt=None,
            overwrite=False,
        )

        downloader._version_dir()

        # Verify config.pbtxt was not created
        config_path = os.path.join(self.temp_dir, model_name, "config.pbtxt")
        self.assertFalse(os.path.exists(config_path))

    def test_multiple_version_dir_calls_are_idempotent(self):
        """Test that calling _version_dir multiple times is safe."""
        model_name = "test-model"
        config_content = "name: test-model"

        downloader = TritonModelDownloader(
            sources=[],
            base_dir=self.temp_dir,
            model_name=model_name,
            config_pbtxt=config_content,
            overwrite=False,
        )

        # Call multiple times
        version_dir1 = downloader._version_dir()
        version_dir2 = downloader._version_dir()
        version_dir3 = downloader._version_dir()

        # All should return the same path
        self.assertEqual(version_dir1, version_dir2)
        self.assertEqual(version_dir2, version_dir3)

        # Directory should still exist
        self.assertTrue(os.path.exists(version_dir1))

    def test_prepare_and_download_with_no_sources(self):
        """Test prepare_and_download with empty sources list."""
        downloader = TritonModelDownloader(
            sources=[],
            base_dir=self.temp_dir,
            model_name="test-model",
            config_pbtxt="name: test",
            overwrite=False,
        )

        result = downloader.prepare_and_download()

        # Should return empty list
        self.assertEqual([], result)
        self.assertIsInstance(result, list)

    def test_prepare_and_download_creates_directory_structure(self):
        """Test that prepare_and_download creates the correct directory structure."""
        model_name = "test-model"
        downloader = TritonModelDownloader(
            sources=[],
            base_dir=self.temp_dir,
            model_name=model_name,
            config_pbtxt="name: test",
            overwrite=False,
        )

        downloader.prepare_and_download()

        # Verify directory structure exists
        model_dir = os.path.join(self.temp_dir, model_name)
        version_dir = os.path.join(model_dir, "1")
        config_path = os.path.join(model_dir, "config.pbtxt")

        self.assertTrue(os.path.exists(model_dir))
        self.assertTrue(os.path.exists(version_dir))
        self.assertTrue(os.path.exists(config_path))

    def test_downloader_with_different_model_names(self):
        """Test downloader with various model names."""
        test_cases = [
            ("simple-model", "Simple model name"),
            ("model-with-dashes", "Model with dashes"),
            ("model_with_underscores", "Model with underscores"),
            ("model123", "Model with numbers"),
            ("UPPERCASE-MODEL", "Uppercase model name"),
        ]

        for model_name, description in test_cases:
            with self.subTest(model_name=model_name, description=description):
                downloader = TritonModelDownloader(
                    sources=[],
                    base_dir=self.temp_dir,
                    model_name=model_name,
                    config_pbtxt=None,
                    overwrite=False,
                )

                downloader.prepare_and_download()

                # Verify directory was created with correct name
                model_dir = os.path.join(self.temp_dir, model_name)
                self.assertTrue(os.path.exists(model_dir))

                # Clean up for next iteration
                shutil.rmtree(model_dir)

    def test_downloader_with_nested_base_dir(self):
        """Test downloader with nested base directory."""
        nested_dir = os.path.join(self.temp_dir, "level1", "level2", "level3")

        downloader = TritonModelDownloader(
            sources=[],
            base_dir=nested_dir,
            model_name="test-model",
            config_pbtxt=None,
            overwrite=False,
        )

        downloader.prepare_and_download()

        # Verify nested structure was created
        expected_model_dir = os.path.join(nested_dir, "test-model")
        expected_version_dir = os.path.join(expected_model_dir, "1")

        self.assertTrue(os.path.exists(expected_model_dir))
        self.assertTrue(os.path.exists(expected_version_dir))
