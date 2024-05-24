import os
import unittest
from unittest import mock

from marqo.api.exceptions import IndexNotFoundError, InvalidArgError
from marqo.tensor_search import tensor_search
from tests.marqo_test import MarqoTestCase
from tests.utils.transition import add_docs_caller


@unittest.skip
class TestBoostFieldScores(MarqoTestCase):

    def setUp(self):
        self.index_name_1 = "my-test-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as e:
            pass
        finally:
            tensor_search.create_vector_index(
                index_name=self.index_name_1, config=self.config)

            add_docs_caller(config=self.config, index_name=self.index_name_1, docs=[
                {
                    "Title": "The Travels of Marco Polo",
                    "Description": "A 13th-century travelogue describing Polo's travels",
                    "_id": "article_590"
                },
                {
                    "Title": "Extravehicular Mobility Unit (EMU)",
                    "Description": "The EMU is a spacesuit that provides environmental protection, "
                                   "mobility, life support, and communications for astronauts",
                    "_id": "article_591"
                }
            ], auto_refresh=True)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self):
        self.device_patcher.stop()

    def test_score_is_boosted(self):
        q = "What is the best outfit to wear on the moon?"

        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q,
        )
        res_boosted = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q, boost={'Title': [5, 1]}
        )
        score = res['hits'][0]['_score']
        score_boosted = res_boosted['hits'][0]['_score']
        self.assertGreater(score_boosted, score)

    def test_boost_empty_dict(self):
        """Passing an empty dict in the boost argument should not affect the score.
        """
        q = "What is the best outfit to wear on the moon?"

        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q
        )
        res_boosted = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q, boost={}
        )

        score = res['hits'][0]['_score']
        score_boosted = res_boosted['hits'][0]['_score']

        self.assertEqual(score_boosted, score)

    def test_different_attributes_searched_and_boosted(self):
        """An error should be raised if the user tries to
        boost a field which is not being searched.
        """
        q = "What is the best outfit to wear on the moon?"

        with self.assertRaises(InvalidArgError) as ctx:
            res_boosted = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text=q,
                searchable_attributes=['Description'], boost={'Title': [0.5, 1]}
            )

        self.assertTrue('Title' in str(ctx.exception))

    def test_boost_boost_varieties(self):
        q = "What is the best outfit to wear on the moon?"
        boosts = [
            {'Title': [5, 1], 'Description': [-1, -4]},
            {'Title': [0, 0], 'Description': [0, 0]},
            {'Title': [5, 1], 'Description': [-1, -4]},
            {'Title': [5], 'Description': [-1, -4]},
            {'Title': [5]},
            {'Description': [-1, -4.4]},
        ]
        for boost in boosts:
            res_boosted = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text=q, boost=boost
            )

@unittest.skip
class TestBoostFieldScoresComparison(MarqoTestCase):

    def setUp(self):
        self.index_name_1 = "my-test-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as e:
            pass

        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self):
        self.device_patcher.stop()

    def test_boost_multiple_fields(self):
        add_docs_caller(config=self.config, index_name=self.index_name_1, docs=[
            {
                "Title": "A comparison of the best pets",
                "Description": "Animals",
                "_id": "d1"
            },
            {
                "Title": "The history of dogs",
                "Description": "A history of household pets",
                "_id": "d2"
            }
        ])

        q = "What are the best pets"

        res_boosted = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q, boost={
                'Title': [5, 1], 'Description': [-1, -4]
            }
        )
        for res in res_boosted['hits']:
            assert list(res['_highlights'].keys())[0] == 'Title'

        res_boosted_inverse = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q, boost={
                'Title': [-1, -4], 'Description': [5, 1]
            }
        )
        for res in res_boosted_inverse['hits']:
            assert list(res['_highlights'].keys())[0] == 'Description'

    def test_boost_equation_single_field(self):
        # add a test to check if the score is boosted as expected
        add_docs_caller(config=self.config, index_name=self.index_name_1, docs=[
            {
                "Title": "A comparison of the best pets",
                "Description": "Animals",
                "_id": "d1"
            },
            {
                "Title": "The history of dogs",
                "Description": "A history of household pets",
                "_id": "d2"
            }
        ], auto_refresh=True)

        q = "What are the best pets"

        res_boosted = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q, searchable_attributes=['Title'], boost={
                'Title': [5, 1]
            }
        )

        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q, searchable_attributes=['Title']
        )

        res_boosted_inverse = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q, searchable_attributes=['Title'],
            boost={
                'Title': [-1, -4]
            }
        )

        score = res['hits'][0]['_score']
        score_boosted = res_boosted['hits'][0]['_score']
        score_inverse = res_boosted_inverse['hits'][1]["_score"]

        self.assertEqual(score * 5 + 1, score_boosted)
        self.assertEqual(score * -1 - 4, score_inverse)

    def test_boost_equation_multiple_fields(self):
        # add a test to check if the score is boosted as expected
        add_docs_caller(config=self.config, index_name=self.index_name_1, docs=[
            {
                "Title": "A comparison of the best pets",
                "Description": "Animals",
                "_id": "d1"
            },
            {
                "Title": "The history of dogs",
                "Description": "A history of household pets",
                "_id": "d2"
            }
        ], auto_refresh=True)

        q = "What are the best pets"

        res_boosted = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q, boost={
                'Title': [5, 1], 'Description': [-1, -1],
            }
        )

        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q,
        )
        score = res['hits'][0]['_score']
        score_boosted = res_boosted['hits'][0]['_score']

        self.assertEqual(score * 5 + 1, score_boosted)

    def test_boost_equation_with_multiple_docs(self):
        # add a test to check if the score is boosted as expected
        for num_of_doc in range(1, 20):
            add_docs_caller(config=self.config, index_name=self.index_name_1, docs=[
                                                                                       {
                                                                                           "Title": "A comparison of the best pets",
                                                                                           "Description": "Animals",
                                                                                       },
                                                                                   ] * num_of_doc, auto_refresh=True)

            q = "What are the best pets"

            res_boosted = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text=q, boost={
                    'Title': [5, 1],
                }
            )

            res = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text=q,
            )

            for regular, boosted in zip(res['hits'], res_boosted['hits']):
                score = regular['_score']
                boosted_score = boosted['_score']
                self.assertEqual(score * 5 + 1, boosted_score)

            try:
                tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
            except IndexNotFoundError as e:
                pass
            finally:
                tensor_search.create_vector_index(
                    index_name=self.index_name_1, config=self.config)

    def test_boost_equation_with_multiple_docs_2(self):
        # add a test to check if the score is boosted as expected
        for num_of_doc in range(1, 20):
            add_docs_caller(config=self.config, index_name=self.index_name_1, docs=[
                                                                                       {
                                                                                           "Title": "A comparison of the best pets",
                                                                                           "Description": "Animals",
                                                                                       },
                                                                                   ] * num_of_doc, auto_refresh=True)

            q = "What are the best pets"

            res_boosted = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text=q, boost={
                    'Title': [1, 1], 'Description': [1, 1]
                }
            )
            res = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text=q,
            )

            for regular, boosted in zip(res['hits'], res_boosted['hits']):
                score = regular['_score']
                boosted_score = boosted['_score']
                self.assertEqual(score * 1 + 1, boosted_score)

            try:
                tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
            except IndexNotFoundError as e:
                pass
            finally:
                tensor_search.create_vector_index(
                    index_name=self.index_name_1, config=self.config)

    def test_boost_equation_with_pagination_docs(self):
        # add a test to check if the score is boosted as expected
        num_of_doc = 50
        add_docs_caller(config=self.config, index_name=self.index_name_1, docs=[
                                                                                   {
                                                                                       "Title": "A comparison of the best pets",
                                                                                       "Description": "Animals",
                                                                                   },
                                                                               ] * num_of_doc, auto_refresh=True)

        q = "What are the best pets"

        res_boosted = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q, boost={
                'Title': [1, 1], "Description": [1, 1],
            }
        )

        res = tensor_search.search(
            config=self.config, index_name=self.index_name_1, text=q, offset=20, searchable_attributes=["Title"],
        )

        for regular, boosted in zip(res['hits'], res_boosted['hits']):
            score = regular['_score']
            boosted_score = boosted['_score']
            self.assertEqual(score * 1 + 1, boosted_score)

    def test_boost_equation_with_different_fields(self):
        # add a test to check if the score is boosted as expected
        num_of_doc = 20
        for num_of_fields in range(3, 10):
            basic_docs = {"Title": "A comparison of the best pets",
                          "Description": "Animals", }
            basic_boost = {
                'Title': [1, 1], "Description": [1, 1],
            }

            docs = basic_docs.copy()
            boost = basic_boost.copy()
            for i in range(1, num_of_fields):
                docs[f"void_field_{i}"] = f"void_value_{i}"
                boost[f"void_field_{i}"] = [1, 1]

            add_docs_caller(config=self.config, index_name=self.index_name_1, docs=[
                                                                                       docs
                                                                                   ] * num_of_doc, auto_refresh=True)

            q = "What are the best pets"

            res_boosted = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text=q, boost=boost
            )

            res = tensor_search.search(
                config=self.config, index_name=self.index_name_1, text=q,
            )

            for regular, boosted in zip(res['hits'], res_boosted['hits']):
                score = regular['_score']
                boosted_score = boosted['_score']
                self.assertEqual(score * 1 + 1, boosted_score)

            try:
                tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
            except IndexNotFoundError as e:
                pass
            finally:
                tensor_search.create_vector_index(
                    index_name=self.index_name_1, config=self.config)
