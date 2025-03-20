import unittest
from typing import List, Union
from unittest.mock import MagicMock

import numpy as np

from marqo.api.exceptions import BadRequestError, InvalidArgError
from marqo.core.inference.api import Inference, Modality, InferenceResult, InferenceRequest, TextPreprocessingConfig, \
    ImagePreprocessingConfig, AudioPreprocessingConfig, VideoPreprocessingConfig, InferenceErrorModel, ModelError, \
    InferenceError
from marqo.exceptions import InternalError
from marqo.tensor_search.models.external_apis.hf import HfAuth
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.search import VectorisedJobs
from marqo.tensor_search.tensor_search import vectorise_jobs


def vectorise_job(modality: Modality, content: List[str]):
    return VectorisedJobs(
        model_name='model',
        model_properties={'a': 'b'},
        content=content,
        device='cpu',
        normalize_embeddings=True,
        media_download_headers={'header1': 'value1'},
        model_auth=ModelAuth(hf=HfAuth(token='a')),
        modality=modality
    )


class TestVectoriseJobs(unittest.TestCase):

    def setUp(self):
        self.inference = MagicMock(spec=Inference)

    def test_vectorise_jobs_should_skip_empty_job_list(self):
        result = vectorise_jobs(self.inference, [])
        self.assertEqual(result, {})
        self.inference.assert_not_called()

    def test_vectorise_jobs_should_skip_job_with_empty_content(self):
        result = vectorise_jobs(self.inference, [vectorise_job(Modality.TEXT, [])])
        self.assertEqual(result, {})
        self.inference.assert_not_called()

    def test_vectorise_jobs_should_construct_inference_requests_for_all_modalities(self):
        test_cases = [
            # modality, expected preprocessing config
            (Modality.TEXT, TextPreprocessingConfig()),
            (Modality.IMAGE, ImagePreprocessingConfig(download_header={'header1': 'value1'}, download_thread_count=1)),
            (Modality.AUDIO, AudioPreprocessingConfig(download_header={'header1': 'value1'}, download_thread_count=1)),
            (Modality.VIDEO, VideoPreprocessingConfig(download_header={'header1': 'value1'}, download_thread_count=1)),
        ]

        for modality, expected_preprocessing in test_cases:
            with self.subTest(modality=modality):
                job = vectorise_job(modality, ['a', 'b'])

                self.inference.reset_mock()
                self.inference.vectorise.side_effect = [
                    InferenceResult(result=[[('a', np.array([1.0, 2.3]))], [('b', np.array([1.0, 2.3]))]])
                ]

                result = vectorise_jobs(self.inference, [job])

                self.assertEqual(1, len(result))
                self.assertIn({'a': [1.0, 2.3], 'b': [1.0, 2.3]}, result.values())

                self.assertEqual(1, self.inference.vectorise.call_count)
                inference_request: InferenceRequest = self.inference.vectorise.call_args_list[0][0][0]

                self.assertEqual(job.modality, inference_request.modality)
                self.assertEqual(job.content, inference_request.contents)
                self.assertEqual(job.model_name, inference_request.model_config.model_name)
                self.assertEqual(job.model_properties, inference_request.model_config.model_properties)
                self.assertEqual(job.model_auth, inference_request.model_config.model_auth)
                self.assertEqual(job.normalize_embeddings, inference_request.model_config.normalize_embeddings)
                self.assertEqual(job.device, inference_request.device)
                self.assertEqual(True, inference_request.use_inference_cache)
                self.assertEqual(False, inference_request.return_individual_error)
                self.assertEqual(expected_preprocessing, inference_request.preprocessing_config)

    def test_vectorise_jobs_should_handle_multiple_jobs(self):
        job1 = vectorise_job(Modality.TEXT, ['a', 'b'])
        job2 = vectorise_job(Modality.IMAGE, ['http://image-url'])

        self.inference.vectorise.side_effect = [
            InferenceResult(result=[[('a', np.array([1.0, 2.3]))], [('b', np.array([1.0, 2.3]))]]),
            InferenceResult(result=[[('http://image-url', np.array([1.0, 2.3]))]]),
        ]

        result = vectorise_jobs(self.inference, [job1, job2])

        self.assertEqual(2, len(result))
        self.assertIn({'a': [1.0, 2.3], 'b': [1.0, 2.3]}, result.values())
        self.assertIn({'http://image-url': [1.0, 2.3]}, result.values())

        self.assertEqual(2, self.inference.vectorise.call_count)
        self.assertEqual(job1.modality, self.inference.vectorise.call_args_list[0][0][0].modality)
        self.assertEqual(job2.modality, self.inference.vectorise.call_args_list[1][0][0].modality)

    def test_vectorise_jobs_should_raise_internal_error_when_embedding_list_does_not_match_content(self):
        job = vectorise_job(Modality.TEXT, ['a', 'b'])

        self.inference.vectorise.side_effect = [
            InferenceResult(result=[[('a', np.array([1.0, 2.3]))]]),
        ]

        with self.assertRaises(InternalError) as context:
            vectorise_jobs(self.inference, [job])
        self.assertEqual('Inference result contains embeddings for 1 query items, but 2 is expected',
                         str(context.exception))

    def test_vectorise_jobs_should_raise_internal_error_when_individual_errors_are_returned(self):
        job = vectorise_job(Modality.TEXT, ['a', 'b'])
        self.inference.vectorise.side_effect = [
            InferenceResult(result=[
                InferenceErrorModel(error_message='an error'),
                [('b', np.array([1.0, 2.3]))]
            ])
        ]

        with self.assertRaises(InternalError) as context:
            vectorise_jobs(self.inference, [job])
        self.assertEqual("Individual errors returned when vectorising query string: ['a: an error']",
                         str(context.exception))

    def test_vectorise_jobs_should_raise_internal_error_when_inference_result_is_chunked(self):
        job = vectorise_job(Modality.TEXT, ['a', 'b'])
        self.inference.vectorise.side_effect = [
            InferenceResult(result=[
                [('a_chunk1', np.array([1.0, 2.3])), ('a_chunk2', np.array([1.0, 2.3]))],
                [('b', np.array([1.0, 2.3]))]
            ])
        ]

        with self.assertRaises(InternalError) as context:
            vectorise_jobs(self.inference, [job])
        self.assertEqual("Tensor query string should not be chunked but some query items "
                         "have multiple chunks: [('a', 2)]",
                         str(context.exception))

    def test_vectorise_jobs_should_raise_bad_request_error_when_model_error_is_raised(self):
        job = vectorise_job(Modality.TEXT, ['a', 'b'])

        self.inference.vectorise.side_effect = ModelError(message='model error')

        with self.assertRaises(BadRequestError) as context:
            vectorise_jobs(self.inference, [job])
        self.assertEqual("BadRequestError: Problem vectorising query. Reason: model error",
                         str(context.exception))

    def test_vectorise_jobs_should_raise_invalid_arg_error_when_other_inference_error_is_raised(self):
        job = vectorise_job(Modality.TEXT, ['a', 'b'])

        self.inference.vectorise.side_effect = InferenceError(message='inference error')

        with self.assertRaises(InvalidArgError) as context:
            vectorise_jobs(self.inference, [job])
        self.assertEqual("InvalidArgError: Error vectorising content: ['a', 'b']. Message: inference error",
                         str(context.exception))
