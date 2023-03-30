import PIL
from marqo.s2_inference import clip_utils, types, random_utils, s2_inference
import unittest
from unittest import mock


class TestVectorise(unittest.TestCase):

    def test_vectorise_in_batches(self):

        mock_model = mock.MagicMock()
        mock_model.encode = mock.MagicMock()

        random_model = random_utils.Random(model_name='mock_model', embedding_dim=128)

        def func(*args,**kwargs):
            return random_model.encode(*args, **kwargs)

        mock_model.encode.side_effect = func
        mock_model_props = {
            "name": "mock_model",
            "dimensions": random_model.embedding_dimension,
            "tokens": 128,
            "type": "sbert"
        }

        mock_available_models = {
            s2_inference._create_model_cache_key(
                model_name='mock_model', device='cpu',
                model_properties=mock_model_props
            ): mock_model
        }

        @mock.patch('marqo.s2_inference.s2_inference.available_models', mock_available_models)
        @mock.patch('marqo.s2_inference.s2_inference._update_available_models', mock.MagicMock())
        def run():
            s2_inference.vectorise(model_name='mock_model', content=['just a single content'],
                                   model_properties=mock_model_props)

            return True
        assert run()

    def test_vectorise_empty_content(self):
        mock_model = mock.MagicMock()
        mock_model.encode = mock.MagicMock()

        random_model = random_utils.Random(model_name='mock_model', embedding_dim=128)

        def func(*args, **kwargs):
            return random_model.encode(*args, **kwargs)

        mock_model.encode.side_effect = func
        mock_model_props = {
            "name": "mock_model",
            "dimensions": random_model.embedding_dimension,
            "tokens": 128,
            "type": "sbert"
        }

        mock_available_models = {
            s2_inference._create_model_cache_key(
                model_name='mock_model', device='cpu',
                model_properties=mock_model_props
            ): mock_model
        }

        @mock.patch('marqo.s2_inference.s2_inference.available_models', mock_available_models)
        @mock.patch('marqo.s2_inference.s2_inference._update_available_models', mock.MagicMock())
        def run():
            try:
                s2_inference.vectorise(model_name='mock_model', content=[],
                                       model_properties=mock_model_props)
            except ValueError as e:
                assert 'received empty input' in str(e)
            return True
        assert run()

    def test_vectorise_in_batches_with_different_batch_sizes(self):
        mock_model = mock.MagicMock()
        mock_model.encode = mock.MagicMock()

        random_model = random_utils.Random(model_name='mock_model', embedding_dim=128)

        def func(*args, **kwargs):
            return random_model.encode(*args, **kwargs)

        mock_model.encode.side_effect = func
        mock_model_props = {
            "name": "mock_model",
            "dimensions": random_model.embedding_dimension,
            "tokens": 128,
            "type": "sbert"
        }

        mock_available_models = {
            s2_inference._create_model_cache_key(
                model_name='mock_model', device='cpu',
                model_properties=mock_model_props
            ): mock_model
        }

        content_list = ['content1', 'content2', 'content3', 'content4', 'content5']

        @mock.patch('marqo.s2_inference.s2_inference.available_models', mock_available_models)
        @mock.patch('marqo.s2_inference.s2_inference._update_available_models', mock.MagicMock())
        # @mock.patch('marqo.tensor_search.utils.read_env_vars_and_defaults', side_effect=[2, 3])
        @mock.patch('marqo.s2_inference.s2_inference.read_env_vars_and_defaults', side_effect=[2, 3, 10])
        def run(mock_read_env_vars_and_defaults):
            # Test with batch size 2
            s2_inference.vectorise(model_name='mock_model', content=content_list,
                                   model_properties=mock_model_props)

            call_args_list = mock_model.encode.call_args_list
            assert len(call_args_list) == 3
            assert call_args_list[0][0][0] == content_list[:2]
            assert call_args_list[1][0][0] == content_list[2:4]
            assert call_args_list[2][0][0] == content_list[4:]

            # Reset mock_model.encode call_args_list for the next test
            mock_model.encode.reset_mock()

            # Test with batch size 3
            s2_inference.vectorise(model_name='mock_model', content=content_list,
                                   model_properties=mock_model_props)

            call_args_list = mock_model.encode.call_args_list
            assert len(call_args_list) == 2
            assert call_args_list[0][0][0] == content_list[:3]
            assert call_args_list[1][0][0] == content_list[3:]

        run()


class TestVectoriseBatching(unittest.TestCase):

    def setUp(self):
        self.mock_model = mock.MagicMock()
        self.mock_model.encode = mock.MagicMock()

        random_model = random_utils.Random(model_name='mock_model', embedding_dim=128)

        def func(*args, **kwargs):
            return random_model.encode(*args, **kwargs)

        self.mock_model.encode.side_effect = func
        self.mock_model_props = {
            "name": "mock_model",
            "dimensions": random_model.embedding_dimension,
            "tokens": 128,
            "type": "sbert"
        }

        self.mock_available_models = {
            s2_inference._create_model_cache_key(
                model_name='mock_model', device='cpu',
                model_properties=self.mock_model_props
            ): self.mock_model
        }

        self.content_list = ['content1', 'content2', 'content3', 'content4', 'content5']

    @mock.patch('marqo.s2_inference.s2_inference.available_models', {})
    @mock.patch('marqo.s2_inference.s2_inference._update_available_models', mock.MagicMock())
    def test_vectorise_single_content_item(self):
        s2_inference.available_models.update(self.mock_available_models)

        single_content = 'just a single content'
        result = s2_inference.vectorise(model_name='mock_model', content=single_content,
                                        model_properties=self.mock_model_props)

        self.mock_model.encode.assert_called_once_with(single_content, normalize=True)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    @mock.patch('marqo.s2_inference.s2_inference.available_models', {})
    @mock.patch('marqo.s2_inference.s2_inference._update_available_models', mock.MagicMock())
    def test_vectorise_varying_content_lengths(self):
        s2_inference.available_models.update(self.mock_available_models)

        varying_length_content = [
            'short',
            'a bit longer content',
            'this content item is quite a bit longer than the others and should be processed correctly'
        ]
        result = s2_inference.vectorise(model_name='mock_model', content=varying_length_content,
                                        model_properties=self.mock_model_props)

        self.assertEqual(self.mock_model.encode.call_count, 1)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(varying_length_content))

    @mock.patch('marqo.tensor_search.utils.read_env_vars_and_defaults')
    def test_vectorise_large_batch_size(self, mock_read_env_vars_and_defaults):
        s2_inference.available_models.update(self.mock_available_models)

        # Set the batch size larger than the number of content items
        large_batch_size = len(self.content_list) + 5
        mock_read_env_vars_and_defaults.return_value = large_batch_size

        # Test with a batch size larger than the number of content items
        s2_inference.vectorise(model_name='mock_model', content=self.content_list,
                               model_properties=self.mock_model_props)

        call_args_list = self.mock_model.encode.call_args_list
        self.assertEqual(len(call_args_list), 1)
        self.assertEqual(call_args_list[0][0][0], self.content_list)

    @mock.patch('marqo.s2_inference.s2_inference.read_env_vars_and_defaults', return_value=1)
    def test_vectorise_batch_size_one(self, mock_read_env_vars_and_defaults):
        s2_inference.available_models.update(self.mock_available_models)

        # Test with a batch size of 1
        s2_inference.vectorise(model_name='mock_model', content=self.content_list,
                               model_properties=self.mock_model_props)

        call_args_list = self.mock_model.encode.call_args_list
        self.assertEqual(len(call_args_list), len(self.content_list))
        for i in range(len(self.content_list)):
            self.assertEqual(call_args_list[i][0][0], [self.content_list[i]])

    def test_vectorise_error_handling(self):
        s2_inference.available_models.update(self.mock_available_models)

        mock_available_models = mock.MagicMock()
        mock_available_models.return_value = self.mock_available_models
        self.mock_model.encode.side_effect = PIL.UnidentifiedImageError('Some error')

        with self.assertRaises(s2_inference.VectoriseError):
            s2_inference.vectorise(model_name='mock_model', content=self.content_list,
                                   model_properties=self.mock_model_props)

