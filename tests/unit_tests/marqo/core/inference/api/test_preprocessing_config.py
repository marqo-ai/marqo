import unittest

from pydantic import ValidationError

from marqo.core.inference.api import ChunkConfig, TextChunkConfig, TextPreprocessingConfig, ImagePreprocessingConfig, \
    AudioPreprocessingConfig, VideoPreprocessingConfig


class TestChunkConfig(unittest.TestCase):

    # Tests for ChunkConfig
    def test_chunk_config_valid(self):
        """Test ChunkConfig with valid input."""
        try:
            config = ChunkConfig(split_length=100, split_overlap=10)
            self.assertEqual(config.split_length, 100)
            self.assertEqual(config.split_overlap, 10)
        except ValidationError:
            self.fail("ChunkConfig raised ValidationError unexpectedly!")

    def test_chunk_config_invalid_split_length(self):
        """Test ChunkConfig with invalid split_length (<= 0)."""
        with self.assertRaises(ValidationError):
            ChunkConfig(split_length=0, split_overlap=10)

    def test_chunk_config_invalid_split_overlap(self):
        """Test ChunkConfig with invalid split_overlap (< 0)."""
        with self.assertRaises(ValidationError):
            ChunkConfig(split_length=100, split_overlap=-5)

    # Tests for TextChunkConfig
    def test_text_chunk_config_valid_split_methods(self):
        """Test TextChunkConfig with split_method."""
        for split_method in ['character', 'word', 'sentence', 'passage']:
            with self.subTest(split_method=split_method):
                try:
                    config = TextChunkConfig(
                        split_length=100,
                        split_overlap=10,
                        split_method=split_method
                    )
                    self.assertEqual(config.split_method, split_method)
                except ValidationError:
                    self.fail(f"TextChunkConfig raised ValidationError unexpectedly for '{split_method}' method!")

    def test_split_length_greater_than_overlap(self):
        """Test that split_length must be greater than split_overlap."""
        with self.assertRaises(ValidationError):
            TextChunkConfig(
                split_length=10,
                split_overlap=20,
                split_method='word'
            )

    def test_text_chunk_config_invalid_split_method(self):
        """Test TextChunkConfig with invalid split_method."""
        with self.assertRaises(ValidationError):
            TextChunkConfig(
                split_length=100,
                split_overlap=10,
                split_method='invalid_method'
            )

    def test_text_chunk_config_missing_split_method(self):
        """Test TextChunkConfig without providing split_method."""
        with self.assertRaises(ValidationError):
            TextChunkConfig(
                split_length=100,
                split_overlap=10
                # split_method is missing
            )

    def test_text_chunk_config_boundary_split_overlap(self):
        """Test TextChunkConfig with split_overlap at boundary value (0)."""
        try:
            config = TextChunkConfig(
                split_length=100,
                split_overlap=0,
                split_method='sentence'
            )
            self.assertEqual(config.split_overlap, 0)
        except ValidationError:
            self.fail("TextChunkConfig raised ValidationError unexpectedly with split_overlap=0!")

    def test_chunk_config_aliases(self):
        """Test ChunkConfig aliases for splitLength and splitOverlap."""
        try:
            config = ChunkConfig(splitLength=150, splitOverlap=15)
            self.assertEqual(config.split_length, 150)
            self.assertEqual(config.split_overlap, 15)
        except ValidationError:
            self.fail("ChunkConfig raised ValidationError unexpectedly when using aliases!")

    def test_text_chunk_config_aliases(self):
        """Test TextChunkConfig aliases including splitMethod."""
        try:
            config = TextChunkConfig(splitLength=200, splitOverlap=20, splitMethod='passage')
            self.assertEqual(config.split_length, 200)
            self.assertEqual(config.split_overlap, 20)
            self.assertEqual(config.split_method, 'passage')
        except ValidationError:
            self.fail("TextChunkConfig raised ValidationError unexpectedly when using aliases!")


class TestTextPreprocessingConfig(unittest.TestCase):
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = TextPreprocessingConfig()
        self.assertFalse(config.should_chunk)
        self.assertIsNone(config.text_prefix)
        self.assertIsNone(config.chunk_config)

    def test_valid_configuration_without_chunking(self):
        """Test configuration when chunking is disabled."""
        config = TextPreprocessingConfig(should_chunk=False, text_prefix="PREFIX")
        self.assertFalse(config.should_chunk)
        self.assertEqual(config.text_prefix, "PREFIX")
        self.assertIsNone(config.chunk_config)

    def test_valid_configuration_with_chunking(self):
        """Test configuration when chunking is enabled with chunk_config provided."""
        chunk_config = TextChunkConfig(
            split_length=100,
            split_overlap=10,
            split_method='word'
        )
        config = TextPreprocessingConfig(
            should_chunk=True,
            text_prefix="PREFIX",
            chunk_config=chunk_config
        )
        self.assertTrue(config.should_chunk)
        self.assertEqual(config.text_prefix, "PREFIX")
        self.assertEqual(config.chunk_config, chunk_config)

    def test_missing_chunk_config_when_should_chunk(self):
        """Test that a ValueError is raised when should_chunk is True but chunk_config is None."""
        with self.assertRaises(ValueError) as context:
            TextPreprocessingConfig(should_chunk=True)
        self.assertIn("`chunk_config` must be provided when `should_chunk` is True.", str(context.exception))

    def test_extra_chunk_config_when_should_not_chunk(self):
        """Test that a ValueError is raised when should_chunk is True but chunk_config is None."""
        with self.assertRaises(ValueError) as context:
            TextPreprocessingConfig(
                should_chunk=False,
                chunk_config=TextChunkConfig(
                    split_length=100,
                    split_overlap=10,
                    split_method='word'
                )
            )
        self.assertIn("`chunk_config` must not be provided when `should_chunk` is False.", str(context.exception))

    def test_alias_fields(self):
        """Test that alias fields are correctly mapped."""
        data = {
            "shouldChunk": True,
            "textPrefix": "Sample Prefix",
            "chunkConfig": {
                "splitLength": 100,
                "splitOverlap": 10,
                "splitMethod": "word"
            }
        }
        config = TextPreprocessingConfig(**data)
        self.assertTrue(config.should_chunk)
        self.assertEqual(config.text_prefix, "Sample Prefix")
        self.assertIsNotNone(config.chunk_config)
        self.assertEqual(config.chunk_config, TextChunkConfig(
            split_length=100,
            split_overlap=10,
            split_method='word'
        ))

    def test_invalid_chunk_config_type(self):
        """Test that providing an invalid type for chunk_config raises a ValidationError."""
        with self.assertRaises(ValidationError):
            TextPreprocessingConfig(
                should_chunk=True,
                chunk_config="invalid_type"  # Should be an instance of TextChunkConfig
            )

    def test_explicit_none_chunk_config(self):
        """Test explicitly setting chunk_config to None with should_chunk=False."""
        config = TextPreprocessingConfig(should_chunk=False, chunk_config=None)
        self.assertFalse(config.should_chunk)
        self.assertIsNone(config.chunk_config)

    def test_immutability(self):
        config = TextPreprocessingConfig()
        with self.assertRaises(TypeError) as context:
            config.should_chunk = True
        self.assertIn('"TextPreprocessingConfig" is immutable and does not support item assignment',
                      str(context.exception))


class TestImagePreprocessingConfig(unittest.TestCase):
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ImagePreprocessingConfig()
        self.assertFalse(config.should_chunk)
        self.assertEqual(3000, config.download_timeout_ms)
        self.assertIsNone(config.download_thread_count)
        self.assertIsNone(config.download_header)
        self.assertIsNone(config.patch_method)

    def test_valid_configuration_without_chunking(self):
        """Test creating config without chunking."""
        config = ImagePreprocessingConfig(
            should_chunk=False,
            download_timeout_ms=5000,
            download_thread_count=4,
            download_header={"Authorization": "Bearer token"},
        )
        self.assertFalse(config.should_chunk)
        self.assertEqual(config.download_timeout_ms, 5000)
        self.assertEqual(config.download_thread_count, 4)
        self.assertEqual(config.download_header, {"Authorization": "Bearer token"})
        self.assertIsNone(config.patch_method)

    def test_valid_configuration_with_chunking(self):
        """Test creating config with chunking and valid patch_method."""
        for patch_method in ['simple', 'frcnn', 'dino-v1', 'dino-v2', 'marqo-yolo']:
            with self.subTest(patch_method=patch_method):
                config = ImagePreprocessingConfig(
                    should_chunk=True,
                    patch_method=patch_method
                )
                self.assertTrue(config.should_chunk)
                self.assertEqual(config.patch_method, patch_method)

    def test_invalid_patch_method(self):
        """Test that an invalid patch_method value raises ValidationError."""
        with self.assertRaises(ValidationError) as context:
            ImagePreprocessingConfig(
                should_chunk=True,
                patch_method="invalid_method"
            )
        self.assertIn("unexpected value; permitted: ", str(context.exception))

    def test_missing_patch_method_with_chunking(self):
        """Test that missing patch_method raises ValidationError when should_chunk is True."""
        with self.assertRaises(ValidationError) as context:
            ImagePreprocessingConfig(
                should_chunk=True
                # patch_method is not provided
            )
        self.assertIn("`patch_method` must be provided when `should_chunk` is True", str(context.exception))

    def test_patch_method_provided_without_chunking(self):
        with self.assertRaises(ValidationError) as context:
            ImagePreprocessingConfig(
                should_chunk=False,
                patch_method='simple'  # patch_method is provided
            )
        self.assertIn("`patch_method` must not be provided when `should_chunk` is False", str(context.exception))

    def test_aliases_are_handled_correctly(self):
        """Test that aliases are correctly parsed."""
        config = ImagePreprocessingConfig(
            shouldChunk=True,
            downloadTimeoutMs=3000,
            downloadThreadCount=2,
            downloadHeader={"Content-Type": "application/json"},
            patchMethod="dino-v1"
        )
        self.assertTrue(config.should_chunk)
        self.assertEqual(config.download_timeout_ms, 3000)
        self.assertEqual(config.download_thread_count, 2)
        self.assertEqual(config.download_header, {"Content-Type": "application/json"})
        self.assertEqual(config.patch_method, "dino-v1")

    def test_immutability(self):
        config = ImagePreprocessingConfig()
        with self.assertRaises(TypeError) as context:
            config.should_chunk = True
        self.assertIn('"ImagePreprocessingConfig" is immutable and does not support item assignment', str(context.exception))


class TestAudioPreprocessingConfig(unittest.TestCase):
    def test_default_values(self):
        config = AudioPreprocessingConfig()
        self.assertFalse(config.should_chunk)
        self.assertIsNone(config.chunk_config)
        self.assertIsNone(config.download_header)
        self.assertIsNone(config.download_thread_count)

    def test_valid_configuration_without_chunking(self):
        config = AudioPreprocessingConfig(
            should_chunk=False,
            download_header={"Authorization": "Bearer token"},
            download_thread_count=4,
        )

        self.assertFalse(config.should_chunk)
        self.assertIsNone(config.chunk_config)
        self.assertEqual(config.download_header, {"Authorization": "Bearer token"})
        self.assertEqual(config.download_thread_count, 4)

    def test_valid_configuration_with_chunking(self):
        config = AudioPreprocessingConfig(
            should_chunk=True,
            download_header={"Authorization": "Bearer token"},
            download_thread_count=4,
            chunk_config=ChunkConfig(
                split_length=10,
                split_overlap=2
            )
        )

        self.assertTrue(config.should_chunk)
        self.assertEqual(config.download_header, {"Authorization": "Bearer token"})
        self.assertEqual(config.download_thread_count, 4)
        self.assertEqual(config.chunk_config, ChunkConfig(split_length=10, split_overlap=2))

    def test_should_not_chunk_with_chunk_config(self):
        """
        Test that a ValidationError is raised when should_chunk is False but chunk_config is provided.
        """
        with self.assertRaises(ValidationError) as context:
            AudioPreprocessingConfig(
                should_chunk=False,
                chunk_config=ChunkConfig(split_length=10, split_overlap=2),
            )
        self.assertIn("`chunk_config` must not be provided when `should_chunk` is False.", str(context.exception))

    def test_should_chunk_without_chunk_config(self):
        """
        Test that a ValidationError is raised when should_chunk is False but chunk_config is provided.
        """
        with self.assertRaises(ValidationError) as context:
            AudioPreprocessingConfig(
                should_chunk=True
            )
        self.assertIn("`chunk_config` must be provided when `should_chunk` is True.", str(context.exception))

    def test_aliases_are_handled_correctly(self):
        """Test that aliases are correctly parsed."""
        config = AudioPreprocessingConfig(
            shouldChunk=True,
            downloadThreadCount=2,
            downloadHeader={"Content-Type": "application/json"},
            chunkConfig=ChunkConfig(splitLength=10, splitOverlap=2),
        )
        self.assertTrue(config.should_chunk)
        self.assertEqual(config.download_thread_count, 2)
        self.assertEqual(config.download_header, {"Content-Type": "application/json"})
        self.assertEqual(config.chunk_config, ChunkConfig(splitLength=10, splitOverlap=2))

    def test_immutability(self):
        config = AudioPreprocessingConfig()
        with self.assertRaises(TypeError) as context:
            config.should_chunk = True
        self.assertIn('"AudioPreprocessingConfig" is immutable and does not support item assignment', str(context.exception))


class TestVideoPreprocessingConfig(unittest.TestCase):
    def test_default_values(self):
        config = VideoPreprocessingConfig()
        self.assertFalse(config.should_chunk)
        self.assertIsNone(config.chunk_config)
        self.assertIsNone(config.download_header)
        self.assertIsNone(config.download_thread_count)

    def test_valid_configuration_without_chunking(self):
        config = VideoPreprocessingConfig(
            should_chunk=False,
            download_header={"Authorization": "Bearer token"},
            download_thread_count=4,
        )

        self.assertFalse(config.should_chunk)
        self.assertIsNone(config.chunk_config)
        self.assertEqual(config.download_header, {"Authorization": "Bearer token"})
        self.assertEqual(config.download_thread_count, 4)

    def test_valid_configuration_with_chunking(self):
        config = VideoPreprocessingConfig(
            should_chunk=True,
            download_header={"Authorization": "Bearer token"},
            download_thread_count=4,
            chunk_config=ChunkConfig(
                split_length=10,
                split_overlap=2
            )
        )

        self.assertTrue(config.should_chunk)
        self.assertEqual(config.download_header, {"Authorization": "Bearer token"})
        self.assertEqual(config.download_thread_count, 4)
        self.assertEqual(config.chunk_config, ChunkConfig(split_length=10, split_overlap=2))

    def test_should_not_chunk_with_chunk_config(self):
        """
        Test that a ValidationError is raised when should_chunk is False but chunk_config is provided.
        """
        with self.assertRaises(ValidationError) as context:
            VideoPreprocessingConfig(
                should_chunk=False,
                chunk_config=ChunkConfig(split_length=10, split_overlap=2),
            )
        self.assertIn("`chunk_config` must not be provided when `should_chunk` is False.", str(context.exception))

    def test_should_chunk_without_chunk_config(self):
        """
        Test that a ValidationError is raised when should_chunk is False but chunk_config is provided.
        """
        with self.assertRaises(ValidationError) as context:
            VideoPreprocessingConfig(
                should_chunk=True
            )
        self.assertIn("`chunk_config` must be provided when `should_chunk` is True.", str(context.exception))

    def test_aliases_are_handled_correctly(self):
        """Test that aliases are correctly parsed."""
        config = VideoPreprocessingConfig(
            shouldChunk=True,
            downloadThreadCount=2,
            downloadHeader={"Content-Type": "application/json"},
            chunkConfig=ChunkConfig(splitLength=10, splitOverlap=2),
        )
        self.assertTrue(config.should_chunk)
        self.assertEqual(config.download_thread_count, 2)
        self.assertEqual(config.download_header, {"Content-Type": "application/json"})
        self.assertEqual(config.chunk_config, ChunkConfig(splitLength=10, splitOverlap=2))

    def test_immutability(self):
        config = VideoPreprocessingConfig()
        with self.assertRaises(TypeError) as context:
            config.should_chunk = True
        self.assertIn('"VideoPreprocessingConfig" is immutable and does not support item assignment', str(context.exception))
