import unittest

from marqo.s2_inference.reranking import rerank
import copy
import numpy as np

class TestRerankingWithModels(unittest.TestCase):

    def setUp(self) -> None:

        pass

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