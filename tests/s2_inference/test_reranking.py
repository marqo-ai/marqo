import unittest

from marqo.s2_inference.reranking.cross_encoders import FormattedResults, ReRankerText
from marqo.s2_inference.reranking.model_utils import DummyModel
from marqo.s2_inference.reranking import rerank
import pandas as pd
import copy

class TestReranking(unittest.TestCase):

    def setUp(self) -> None:

        pass

    
    def test_fill_ids(self):
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


        assert '_id' not in results_lexical['hits'][0]
        assert '_id' in results_lexical['hits'][1]
        assert '_id' in results_lexical['hits'][2]

        assert '_rerank_id' not in results_lexical['hits'][0]
        assert '_rerank_id' not in results_lexical['hits'][1]
        assert '_rerank_id' not in results_lexical['hits'][2]

        results_lexical_formatted = FormattedResults(results_lexical)        

        assert '_rerank_id' in results_lexical['hits'][0]
        assert '_rerank_id' in results_lexical['hits'][1]
        assert '_rerank_id' in results_lexical['hits'][2]

        assert '_id' not in results_lexical['hits'][0]
        assert '_id' in results_lexical['hits'][1]
        assert '_id' in results_lexical['hits'][2]


    def test_results_to_df(self):

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

        results_lexical_formatted = FormattedResults(results_lexical)

        assert isinstance(results_lexical_formatted.results_df, pd.DataFrame)

        columns = sorted(results_lexical_formatted.results_df.columns.tolist())

        # check the columns converted correctly
        assert columns == ['_highlights', '_id', '_rerank_id', '_score', 'attributes', 'location', 'other']

        # we didn;t lose anything
        assert len(results_lexical_formatted.results_df) == len(results_lexical['hits'])

        # we use a temporary id if none is present
        assert sorted(results_lexical_formatted.results_df._id.apply(str).tolist()) !=  sorted(results_lexical_formatted.results_df._rerank_id.apply(str).tolist())

        # searchable fields if not specified should be all non marqo fields
        assert results_lexical_formatted.searchable_fields == ['attributes', 'location', 'other']

    def test_formatting_for_model(self):

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

        results_lexical_formatted = FormattedResults(results_lexical)
        query = 'hello'

        # this is used as inputs to a model
        inputs_df = results_lexical_formatted.format_for_model(results_lexical_formatted.results_df, results_lexical_formatted.searchable_fields, query=query)
    
        # we verify the size by taking the number of searchablke attributes across all docs (=3) time the number of docs (=3) minus the ones that were not present
        # = 2 since [1] and [2] are each missing a field 
        assert len(inputs_df) == (len(results_lexical_formatted.searchable_fields)*len(results_lexical['hits']) - 2)

        # check the query
        assert set(inputs_df['query'].to_list()) == set([query])

        # check the field content 
        assert all(inputs_df['field_content'].notna()) and all(inputs_df['field_content'].notnull())

        # check the original field names are part of the available fields
        available_fields = ['attributes', 'location', 'other']
        assert sorted(available_fields) == sorted(inputs_df['original_field_name'].unique().tolist())

        # check the model inputs
        rr = ReRankerText('_testing', 'cpu')
        rr.load_model()

        # check the exploded content
        inputs_df_exploded = rr.explode_nested_content_field(inputs_df)        

        assert len(inputs_df) <= len(inputs_df_exploded)

        # check that the extracted content is a subset of the original
        # if the split processing for text changes, there is a small chance this test could fail due to whitespace or similar small difference
        # if this fails, check those things
        for i,row in inputs_df_exploded.iterrows():
            assert row['field_content'] in row['field_content_original']


    def test_reranking_text(self):

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

        rr = ReRankerText('_testing', 'cpu')

        # test the split params
        assert isinstance(rr.split_length, int)
        assert isinstance(rr.split_overlap, int)
        assert isinstance(rr.split_method, str)

        # loading the model
        assert rr.model == None
        rr.load_model()
        assert isinstance(rr.model, DummyModel)

        # preparing model inputs
        results_lexical_formatted = FormattedResults(results_lexical)
        query = 'hello'

        rr.rerank(query, results_lexical)

        assert len(results_lexical['hits']) == len(results_lexical_copy['hits'])

        assert all( doc['_highlights'] == [] for doc in results_lexical_copy['hits'])

        assert all( doc['_highlights'] == [] for doc in results_lexical['hits'])

        assert all( len(doc['_reranked_highlights']) >= 1 for doc in results_lexical['hits'])

        assert all( isinstance(doc['_reranked_score'], (int, float)) for doc in results_lexical['hits'])

        results_lexical_reranked = copy.deepcopy(results_lexical)

        # this cleans up the reranked fields to make it 
        rerank.cleanup_final_reranked_results(results_lexical)

        assert all( '_reranked_highlights' not in doc for doc in results_lexical['hits'])

        assert all( '_reranked_score' not in doc for doc in results_lexical['hits'])


    def test_rerank(self):

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
        model_name = '_testing'
        rerank.rerank_search_results(results_lexical, 'hello', model_name, 'cpu')
        
        assert len(results_lexical['hits']) == len(results_lexical_copy['hits'])

        assert all( doc['_highlights'] == [] for doc in results_lexical_copy['hits'])

        assert all( doc['_highlights'] != [] for doc in results_lexical['hits'])

        assert all( len(doc['_highlights']) >= 1 for doc in results_lexical['hits'])

        assert all( isinstance(doc['_score'], (int, float)) for doc in results_lexical['hits'])

        # check monotinicity of scores
        all_scores = [doc['_score'] for doc in results_lexical['hits']]
        assert all( s1 >= s2  for s1,s2 in zip(all_scores[:-1], all_scores[1:]))