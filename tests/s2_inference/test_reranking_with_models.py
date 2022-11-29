import unittest
import copy
import numpy as np

from marqo.s2_inference.reranking import rerank
from marqo.s2_inference.errors import RerankerError,RerankerNameError
from marqo.s2_inference.s2_inference import clear_loaded_models


class TestRerankingWithModels(unittest.TestCase):

    def setUp(self) -> None:

        pass

    def tearDown(self) -> None:
        clear_loaded_models()

    def test_reranking_text_sbert(self):

        results_lexical = {'hits': 
                    [{'attributes': 'yello head. pruple shirt. black sweater.',
                        'location': 'images/1.png',
                        'other': 'some other text',
                        # this one has no id
                        '_score': 1.4017934,
                        '_highlights': [],
                        },
                        {'attributes': 'face is viking. body is white turtleneck. background is pearl',
                        # missing locations
                        'other': 'some more text',
                        '_id': 'QmRR6PBkgCdhiSYBM3AY3EWhn4ZbeR2X8Ygpy2veLkcPC5',
                        '_score': 0.2876821,
                        '_highlights': [],
                        },
                        # this one has less fields
                        {'attributes': 'face is bowlcut. body is blue . background is grey. head is tan',
                        'location': 'images/10.png',
                        '_id': 'QmTVYuULK1Qbzh21Y3hzeTFny5AGUSUGAXoGjLqNB2b1at',
                        '_score': 0.2876821,
                        '_highlights': [],
                        }],
                        'processingTimeMs': 49,
                        'query': 'yellow turtleneck',
                        'limit': 10}

        results_lexical_copy = copy.deepcopy(results_lexical)
        model_name = 'cross-encoder/ms-marco-TinyBERT-L-2'
        rerank.rerank_search_results(results_lexical, 'hello', model_name, 'cpu')

        assert len(results_lexical['hits']) == len(results_lexical_copy['hits'])

        assert all( doc['_highlights'] == [] for doc in results_lexical_copy['hits'])

        assert all( doc['_highlights'] != [] for doc in results_lexical['hits'])

        assert all( len(doc['_highlights']) >= 1 for doc in results_lexical['hits'])

        assert all( isinstance(doc['_score'], (int, float, np.float32)) for doc in results_lexical['hits'])

    def test_reranking_text_onnx(self):

        results_lexical = {'hits': 
                    [{'attributes': 'yello head. pruple shirt. black sweater.',
                        'location': 'images/1.png',
                        'other': 'some other text',
                        # this one has no id
                        '_score': 1.4017934,
                        '_highlights': [],
                        },
                        {'attributes': 'face is viking. body is white turtleneck. background is pearl',
                        # missing locations
                        'other': 'some more text',
                        '_id': 'QmRR6PBkgCdhiSYBM3AY3EWhn4ZbeR2X8Ygpy2veLkcPC5',
                        '_score': 0.2876821,
                        '_highlights': [],
                        },
                        # this one has less fields
                        {'attributes': 'face is bowlcut. body is blue . background is grey. head is tan',
                        'location': 'images/10.png',
                        '_id': 'QmTVYuULK1Qbzh21Y3hzeTFny5AGUSUGAXoGjLqNB2b1at',
                        '_score': 0.2876821,
                        '_highlights': [],
                        }],
                        'processingTimeMs': 49,
                        'query': 'yellow turtleneck',
                        'limit': 10}

        results_lexical_copy = copy.deepcopy(results_lexical)
        model_name = 'onnx/cross-encoder/ms-marco-TinyBERT-L-2'
        rerank.rerank_search_results(results_lexical, 'hello', model_name, 'cpu')

        assert len(results_lexical['hits']) == len(results_lexical_copy['hits'])

        assert all( doc['_highlights'] == [] for doc in results_lexical_copy['hits'])

        assert all( doc['_highlights'] != [] for doc in results_lexical['hits'])

        assert all( len(doc['_highlights']) >= 1 for doc in results_lexical['hits'])

        assert all( isinstance(doc['_score'], (int, float, np.float32)) for doc in results_lexical['hits'])

    def test_reranking_onnx_sbert_parity(self):

        results_lexical = {'hits': 
                    [{'attributes': 'yello head. pruple shirt. black sweater.',
                        'location': 'images/1.png',
                        'other': 'some other text',
                        # this one has no id
                        '_score': 1.4017934,
                        '_highlights': [],
                        },
                        {'attributes': 'face is viking. body is white turtleneck. background is pearl',
                        # missing locations
                        'other': 'some more text',
                        '_id': 'QmRR6PBkgCdhiSYBM3AY3EWhn4ZbeR2X8Ygpy2veLkcPC5',
                        '_score': 0.2876821,
                        '_highlights': [],
                        },
                        # this one has less fields
                        {'attributes': 'face is bowlcut. body is blue . background is grey. head is tan',
                        'location': 'images/10.png',
                        '_id': 'QmTVYuULK1Qbzh21Y3hzeTFny5AGUSUGAXoGjLqNB2b1at',
                        '_score': 0.2876821,
                        '_highlights': [],
                        }],
                        'processingTimeMs': 49,
                        'query': 'yellow turtleneck',
                        'limit': 10}

        results_lexical_sbert = copy.deepcopy(results_lexical)
        results_lexical_onnx = copy.deepcopy(results_lexical)

        model_name = 'onnx/cross-encoder/ms-marco-TinyBERT-L-2'
        rerank.rerank_search_results(results_lexical_onnx, 'hello', model_name, 'cpu')

        model_name = 'cross-encoder/ms-marco-TinyBERT-L-2'
        rerank.rerank_search_results(results_lexical_sbert, 'hello', model_name, 'cpu')

        searchable_attributes = ['attributes', 'location', 'other']

        # check the scores
        for ind in range(len(results_lexical['hits'])):
            onnx = results_lexical_onnx['hits'][ind]
            sbert = results_lexical_sbert['hits'][ind]

            for att in searchable_attributes:
                if att in onnx:
                    assert att in sbert
                    assert onnx[att] == sbert[att]
                else:
                    assert att not in sbert

            assert abs(onnx['_score'] - sbert['_score']) < 1e-6

    def test_reranking_images_owl_inconsistent(self):
        # not all results have the searchable filed to rerank over
        results_lexical = {'hits': 
                    [{'attributes': 'yello head. pruple shirt. black sweater.',
                        'location': 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png',
                        'other': 'some other text',
                        # this one has no id
                        '_score': 1.4017934,
                        '_highlights': [],
                        },
                        {'attributes': 'face is viking. body is white turtleneck. background is pearl',
                        # missing locations
                        'other': 'some more text',
                        '_id': 'QmRR6PBkgCdhiSYBM3AY3EWhn4ZbeR2X8Ygpy2veLkcPC5',
                        '_score': 0.2876821,
                        '_highlights': [],
                        },
                        # this one has less fields
                        {'attributes': 'face is bowlcut. body is blue . background is grey. head is tan',
                        'location': 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png',
                        '_id': 'QmTVYuULK1Qbzh21Y3hzeTFny5AGUSUGAXoGjLqNB2b1at',
                        '_score': 0.2876821,
                        '_highlights': [],
                        }],
                        'processingTimeMs': 49,
                        'query': 'yellow turtleneck',
                        'limit': 10}

        results_lexical_copy = copy.deepcopy(results_lexical)
        model_name = "google/owlvit-base-patch32"
        image_location = 'location'
        try:
            rerank.rerank_search_results(search_result=results_lexical, query='hippo', model_name=model_name, 
                            device='cpu')
        except Exception as e:
            assert 'found searchable_attributes=None' in str(e)

        rerank.rerank_search_results(search_result=results_lexical, query='hippo', model_name=model_name, 
                            device='cpu', searchable_attributes=[image_location])

        # for owl, if the document does not have the searchable_field it is removed from the reranked results
        N_out = len([d for d in results_lexical['hits'] if image_location in d])
        N_in = len([d for d in results_lexical_copy['hits'] if image_location in d])

        assert N_in == 2

        assert N_out == N_in

        assert all( doc['_highlights'] == [] for doc in results_lexical_copy['hits'])

        assert all( doc['_highlights'] != [] for doc in results_lexical['hits'])

        assert all( len(doc['_highlights']) >= 1 for doc in results_lexical['hits'])

        assert all( isinstance(doc['_score'], (int, float, np.float32)) for doc in results_lexical['hits'])

    def test_reranking_images_incorrect_model(self):
        # not all results have the searchable filed to rerank over
        results_lexical = {'hits': 
                    [{'attributes': 'yello head. pruple shirt. black sweater.',
                        'location': 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png',
                        'other': 'some other text',
                        # this one has no id
                        '_score': 1.4017934,
                        '_highlights': [],
                        },
                        {'attributes': 'face is viking. body is white turtleneck. background is pearl',
                        # missing locations
                        'other': 'some more text',
                        '_id': 'QmRR6PBkgCdhiSYBM3AY3EWhn4ZbeR2X8Ygpy2veLkcPC5',
                        '_score': 0.2876821,
                        '_highlights': [],
                        },
                        # this one has less fields
                        {'attributes': 'face is bowlcut. body is blue . background is grey. head is tan',
                        'location': 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png',
                        '_id': 'QmTVYuULK1Qbzh21Y3hzeTFny5AGUSUGAXoGjLqNB2b1at',
                        '_score': 0.2876821,
                        '_highlights': [],
                        }],
                        'processingTimeMs': 49,
                        'query': 'yellow turtleneck',
                        'limit': 10}

        results_lexical_copy = copy.deepcopy(results_lexical)
        model_name = "google/owlvt-base-patch32"
        image_location = 'location'
        try:
            rerank.rerank_search_results(search_result=results_lexical, query='hippo', model_name=model_name, 
                            device='cpu', searchable_attributes=[image_location])
        except RerankerError as e:
            assert "could not find model_name=" in str(e)
            #print(e)

    def test_reranking_images_owl_inconsistent_highlights(self):
        # not all results have the searchable filed to rerank over
        results_lexical = {'hits': 
                    [{'attributes': 'yello head. pruple shirt. black sweater.',
                        'location': 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png',
                        'other': 'some other text',
                        # this one has no id
                        '_score': 1.4017934,
                        '_highlights': {"location":[0,0,10,10]},
                        },
                        {'attributes': 'face is viking. body is white turtleneck. background is pearl',
                        # missing locations
                        'other': 'some more text',
                        '_id': 'QmRR6PBkgCdhiSYBM3AY3EWhn4ZbeR2X8Ygpy2veLkcPC5',
                        '_score': 0.2876821,
                        '_highlights': [],
                        },
                        # this one has less fields
                        {'attributes': 'face is bowlcut. body is blue . background is grey. head is tan',
                        'location': 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png',
                        '_id': 'QmTVYuULK1Qbzh21Y3hzeTFny5AGUSUGAXoGjLqNB2b1at',
                        '_score': 0.2876821,
                        '_highlights': {"location":[0,0,20,30]},
                        }],
                        'processingTimeMs': 49,
                        'query': 'yellow turtleneck',
                        'limit': 10}

        results_lexical_copy = copy.deepcopy(results_lexical)
        model_name = "google/owlvit-base-patch32"
        image_location = 'location'
        try:
            rerank.rerank_search_results(search_result=results_lexical, query='hippo', model_name=model_name, 
                            device='cpu', searchable_attributes=None)
        except RerankerError as e:
            pass

        rerank.rerank_search_results(search_result=results_lexical, query='hippo', model_name=model_name, 
                            device='cpu', searchable_attributes=[image_location])

        # for owl, if the document does not have the searchable_field it is removed from the results
        N_out = len([d for d in results_lexical['hits'] if image_location in d])
        N_in = len([d for d in results_lexical_copy['hits'] if image_location in d])

        assert N_in == 2

        assert N_out == N_in

        assert sum( doc['_highlights'] == [] for doc in results_lexical_copy['hits']) == 1

        assert sum( doc['_highlights'] != [] for doc in results_lexical_copy['hits']) == 2
        assert sum( doc['_highlights'] != [] for doc in results_lexical['hits']) == 2

        assert all( len(doc['_highlights']) >= 1 for doc in results_lexical['hits'])

        assert all( isinstance(doc['_score'], (int, float, np.float32)) for doc in results_lexical['hits'])

    def test_reranking_images_owl_consistent(self):
        # all results have the searchable field to rerank over
        results_lexical = {'hits': 
                    [{'attributes': 'yello head. pruple shirt. black sweater.',
                        'location': 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png',
                        'other': 'some other text',
                        # this one has no id
                        '_score': 1.4017934,
                        '_highlights': [],
                        },
                        {'attributes': 'face is viking. body is white turtleneck. background is pearl',
                        # missing locations
                        'other': 'some more text',
                        'location': 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png',
                        '_id': 'QmRR6PBkgCdhiSYBM3AY3EWhn4ZbeR2X8Ygpy2veLkcPC5',
                        '_score': 0.2876821,
                        '_highlights': [],
                        },
                        # this one has less fields
                        {'attributes': 'face is bowlcut. body is blue . background is grey. head is tan',
                        'location': 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png',
                        '_id': 'QmTVYuULK1Qbzh21Y3hzeTFny5AGUSUGAXoGjLqNB2b1at',
                        '_score': 0.2876821,
                        '_highlights': [],
                        }],
                        'processingTimeMs': 49,
                        'query': 'yellow turtleneck',
                        'limit': 10}

        results_lexical_copy = copy.deepcopy(results_lexical)
        model_name = "google/owlvit-base-patch32"
        image_location = 'location'
        try:
            rerank.rerank_search_results(search_result=results_lexical, query='hippo', model_name=model_name, 
                            device='cpu')
        except RerankerError as e:
            pass

        rerank.rerank_search_results(search_result=results_lexical, query='hippo', model_name=model_name, 
                            device='cpu', searchable_attributes=[image_location])

        # for owl, if the document does not have the searchable_field it is removed from the results
        N_out = len([d for d in results_lexical['hits'] if image_location in d])
        N_in = len([d for d in results_lexical_copy['hits'] if image_location in d])

        assert N_in == 3

        assert N_out == N_in

        assert all( doc['_highlights'] == [] for doc in results_lexical_copy['hits'])

        assert all( doc['_highlights'] != [] for doc in results_lexical['hits'])

        assert all( len(doc['_highlights']) >= 1 for doc in results_lexical['hits'])

        assert all( isinstance(doc['_score'], (int, float, np.float32)) for doc in results_lexical['hits'])

    def test_reranking_check_search_has_fields(self):

        results_lexical = {'hits': 
                    [{'attributes': 'yello head. pruple shirt. black sweater.',
                        'location': 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png',
                        'other': 'some other text',
                        # this one has no id
                        '_score': 1.4017934,
                        '_highlights': [],
                        },
                        {'attributes': 'face is viking. body is white turtleneck. background is pearl',
                        # missing locations
                        'other': 'some more text',
                        'location': 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png',
                        '_id': 'QmRR6PBkgCdhiSYBM3AY3EWhn4ZbeR2X8Ygpy2veLkcPC5',
                        '_score': 0.2876821,
                        '_highlights': [],
                        },
                        # this one has less fields
                        {'attributes': 'face is bowlcut. body is blue . background is grey. head is tan',
                        'location': 'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png',
                        '_id': 'QmTVYuULK1Qbzh21Y3hzeTFny5AGUSUGAXoGjLqNB2b1at',
                        '_score': 0.2876821,
                        '_highlights': [],
                        }],
                        'processingTimeMs': 49,
                        'query': 'yellow turtleneck',
                        'limit': 10}

        in_results_field = 'location'

        not_in_results_field = 'location_location'

        assert any(True if in_results_field in r else False for r in results_lexical['hits'])

        assert not any(True if not_in_results_field in r else False for r in results_lexical['hits'])

        assert rerank._check_searchable_fields_in_results(search_results=results_lexical, searchable_fields=None)

        assert rerank._check_searchable_fields_in_results(search_results=results_lexical, searchable_fields=[in_results_field])

        assert not rerank._check_searchable_fields_in_results(search_results=results_lexical, searchable_fields=[not_in_results_field])

        assert not rerank._check_searchable_fields_in_results(search_results=results_lexical, searchable_fields=[not_in_results_field])

        assert not rerank._check_searchable_fields_in_results(search_results=results_lexical, searchable_fields=not_in_results_field)

        assert not rerank._check_searchable_fields_in_results(search_results=results_lexical, searchable_fields=in_results_field)

    def test_reranking_images_owl_empty(self):
        # all results have none of the the searchable field to rerank over
        results_lexical = {'hits': 
                    [{'attributes': 'yello head. pruple shirt. black sweater.',
                       
                        'other': 'some other text',
                        # this one has no id
                        '_score': 1.4017934,
                        '_highlights': [],
                        },
                        {'attributes': 'face is viking. body is white turtleneck. background is pearl',
                        # missing locations
                        'other': 'some more text',
                        
                        '_id': 'QmRR6PBkgCdhiSYBM3AY3EWhn4ZbeR2X8Ygpy2veLkcPC5',
                        '_score': 0.2876821,
                        '_highlights': [],
                        },
                        # this one has less fields
                        {'attributes': 'face is bowlcut. body is blue . background is grey. head is tan',
                        
                        '_id': 'QmTVYuULK1Qbzh21Y3hzeTFny5AGUSUGAXoGjLqNB2b1at',
                        '_score': 0.2876821,
                        '_highlights': [],
                        }],
                        'processingTimeMs': 49,
                        'query': 'yellow turtleneck',
                        'limit': 10}

        results_lexical_copy = copy.deepcopy(results_lexical)
        model_name = "google/owlvit-base-patch32"
        image_location = 'location_location'
       

        rerank.rerank_search_results(search_result=results_lexical, query='hippo', model_name=model_name, 
                            device='cpu', searchable_attributes=[image_location])

        assert results_lexical == results_lexical_copy

        assert all( doc['_highlights'] == [] for doc in results_lexical_copy['hits'])

        assert all( doc['_highlights'] == [] for doc in results_lexical['hits'])

        # since there are no results with the searchable fields the highlights should be the same
        assert all( len(doc['_highlights']) == 0 for doc in results_lexical['hits'])

        assert all( isinstance(doc['_score'], (int, float, np.float32)) for doc in results_lexical['hits'])