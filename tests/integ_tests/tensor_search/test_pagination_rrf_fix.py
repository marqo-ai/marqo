import os
from unittest import mock

import pytest

from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.hybrid_parameters import RetrievalMethod, RankingMethod, HybridParameters
from marqo.core.models.marqo_index import *
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.score_modifiers_object import ScoreModifierLists, ScoreModifierOperator
from tests.integ_tests.marqo_test import MarqoTestCase


class TestRRFPaginationPartialFix(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Create only unstructured index for now
        index_request_unstructured = cls.unstructured_marqo_index_request(
            model=Model(name='hf/e5-base-v2')
        )

        cls.indexes = cls.create_indexes([
            index_request_unstructured
        ])

        cls.index_unstructured = cls.indexes[0]

    def setUp(self) -> None:
        super().setUp()
        # Any tests that call add_document, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

        # Create and add curated test documents
        r = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.index_unstructured.name,
                docs=(self.create_curated_test_data()),
                tensor_fields=['title']
            )
        ).dict(exclude_none=True, by_alias=True)
        self.assertFalse(r['errors'], "Errors in add documents call")

        self.hp_rrf = HybridParameters(
            alpha=0.7,
            rrfK=60,
            retrievalMethod=RetrievalMethod.Disjunction,
            rankingMethod=RankingMethod.RRF,
            queryTensor='machine learning',
            queryLexical='EXACT QUERY',
            # verbose=True,
        )

    def tearDown(self):
        self.device_patcher.stop()

    def create_curated_test_data(self):
        """
        Create simplified test data with deterministic rankings.
        - 5 docs only match tensor (doc_T1 to doc_T5, ranked T1 > T2 > T3 > T4 > T5)
        - 5 docs only match lexical (doc_L1 to doc_L5, ranked L1 > L2 > L3 > L4 > L5)
        - 5 docs match both (doc_B1 to doc_B5, different rankings in tensor vs lexical)
        """
        docs = []

        # 5 docs that only match TENSOR search (semantic similarity)
        # Avoid words "EXACT", "QUERY", "machine", "learning" to prevent lexical matches
        tensor_terms = [
            "artificial intelligence algorithms",     # T1 - highest tensor relevance
            "deep neural systems",                    # T2
            "neural network models",                  # T3
            "computational intelligence frameworks",  # T4
            "data science tools"                      # T5 - lowest tensor relevance
        ]

        for i in range(5):
            doc = {
                "_id": f"doc_T{i+1}",
                "title": f"{tensor_terms[i]} tensor_document_{i+1}",
                "doc_type": "tensor_only",
                "score_boost": i+1,   # the boost should reverse the order
                f"tensor_score_boost_{tensor_terms[i].split()[0]}": 1,  # score modifier to change ranking of tensor search
            }
            docs.append(doc)

        # 5 docs that only match LEXICAL search (exact keyword matches)
        for i in range(5):
            # Use decreasing keyword repetition for deterministic TF scores
            repeated_keywords = " ".join(["EXACT", "QUERY"] * (5 - i))  # More repetition = higher score
            doc = {
                "_id": f"doc_L{i+1}",
                "title": f"{repeated_keywords} lexical_document_{i+1}",
                "doc_type": "lexical_only",
                "score_boost": i+6,  # the boost should reverse the order, and this is higher than tensor match
            }
            docs.append(doc)

        # 5 docs that match BOTH tensor and lexical (for RRF fusion testing)
        # These have DIFFERENT rankings in tensor vs lexical to test RRF properly
        both_docs = [
            # (tensor_keywords, lexical_keywords, tensor_rank, lexical_rank)
            ("machine learning algorithms", " ".join(["EXACT", "QUERY"] * 10), 1, 1),  # B1: high in both
            ("machine learning courses", "EXACT QUERY", 2, 4),                           # B2: high tensor, low lexical
            ("computational models", "EXACT QUERY EXACT QUERY", 3, 2),          # B3: mid tensor, high lexical
            ("data science", "EXACT", 4, 5),                                    # B4: low tensor, lowest lexical
            ("algorithmic systems", "EXACT QUERY EXACT", 5, 3),                 # B5: lowest tensor, mid lexical
        ]

        for i, (tensor_term, lexical_term, tensor_rank, lexical_rank) in enumerate(both_docs):
            doc = {
                "_id": f"doc_B{i+1}",
                "title": f"{lexical_term} {tensor_term} both_document_{i+1}",
                "doc_type": "both",
                "score_boost": i+11,  # the boost should reverse the order, and this is higher than tensor and lexical match
            }
            docs.append(doc)

        return docs

    @pytest.mark.skip_for_multinode
    def test_disjunction_rrf_pagination(self):
        """
        Test that disjunction+RRF pagination produces no duplicates between pages.
        Verifies ranking consistency and proper pagination behavior.

        Lexical search score:         Tensor search score:
        doc_B1: 1.5657032529354344    doc_B2: 0.8774912556666549
        doc_L1: 1.5310213335683778    doc_B1: 0.8624316166639774
        doc_L2: 1.4957020935493452    doc_T1: 0.8551202270206154
        doc_L3: 1.4403238705575117    doc_T2: 0.8550448320679322
        doc_L4: 1.3410214926606240    doc_T3: 0.8523972952471728
        doc_B3: 1.2495362288065737    doc_T4: 0.8500947166713401
        doc_L5: 1.1111902054958356    doc_B4: 0.8500164585022059
        doc_B5: 1.0981049722627436    doc_T5: 0.8471093151792289
        doc_B2: 0.9400916280884359    doc_B3: 0.8368122835299188
        doc_B4: 0.4681934225728979    doc_B5: 0.8334305830362274
                                      doc_L1: 0.8316111466678087
                                      doc_L5: 0.8288859450924600
                                      doc_L3: 0.8269666625377193
                                      doc_L4: 0.8263098149586303
                                      doc_L2: 0.8257278463225931

        Pagination with page size 5: with alpha=0.7, rrf_k=60
        tensor rrf score = alpha * (1.0 / (rank + k))
        lexical rrf score = (1-alpha) * (1.0 / (rank + k))

        ======================================================
        Page 1:
        Lexical search rrf score:     Tensor search rrf score:
        doc_B1: 0.004918              doc_B2: 0.011475
        doc_L1: 0.004839              doc_B1: 0.011290
        doc_L2: 0.004762              doc_T1: 0.011111
        doc_L3: 0.004688              doc_T2: 0.010938
        doc_L4: 0.004615              doc_T3: 0.010769

        After fusion:
        doc_B1: 0.016208   <- In both, highest ranking
        doc_B2: 0.011475
        doc_T1: 0.011111
        doc_T2: 0.010938
        doc_T3: 0.010769
        doc_L1: 0.004839   <- Trimmed off from here
        doc_L2: 0.004762
        doc_L3: 0.004688
        doc_L4: 0.004615

        =======================================================
        Page 2:
        Lexical search rrf score:     Tensor search rrf score:
        doc_B1: 0.004918              doc_B2: 0.011475
        doc_L1: 0.004839              doc_B1: 0.011290
        doc_L2: 0.004762              doc_T1: 0.011111
        doc_L3: 0.004688              doc_T2: 0.010938
        doc_L4: 0.004615              doc_T3: 0.010769
        doc_B3: 0.004545              doc_T4: 0.010606
        doc_L5: 0.004478              doc_B4: 0.010448
        doc_B5: 0.004412              doc_T5: 0.010294
        doc_B2: 0.004348              doc_B3: 0.010145
        doc_B4: 0.004286              doc_B5: 0.01

        After fusion:
        doc_B1: 0.016208
        doc_B2: 0.015823
        doc_B4: 0.014734   (MISSED)
        doc_B3: 0.014690   (MISSED)
        doc_B5: 0.014412   (MISSED)
        doc_T1: 0.011111   <- Start from here  (DUP)
        doc_T2: 0.010938   (DUP)
        doc_T3: 0.010769   (DUP)
        doc_T4: 0.010606
        doc_T5: 0.010294
        doc_L1: 0.004839   <- Trimmed off from here
        doc_L2: 0.004762
        doc_L3: 0.004688
        doc_L4: 0.004615
        doc_L5: 0.004478

        =======================================================
        Page 3:
        Lexical search rrf score:     Tensor search rrf score:
        doc_B1: 0.004918              doc_B2: 0.011475
        doc_L1: 0.004839              doc_B1: 0.011290
        doc_L2: 0.004762              doc_T1: 0.011111
        doc_L3: 0.004688              doc_T2: 0.010938
        doc_L4: 0.004615              doc_T3: 0.010769
        doc_B3: 0.004545              doc_T4: 0.010606
        doc_L5: 0.004478              doc_B4: 0.010448
        doc_B5: 0.004412              doc_T5: 0.010294
        doc_B2: 0.004348              doc_B3: 0.010145
        doc_B4: 0.004286              doc_B5: 0.01
                                      doc_L1: 0.009859
                                      doc_L5: 0.009722
                                      doc_L3: 0.009589
                                      doc_L4: 0.009459
                                      doc_L2: 0.009333

        After fusion:
        doc_B1: 0.016208
        doc_B2: 0.015823
        doc_B4: 0.014734    (MISSED)
        doc_L1: 0.014698    (MISSED)
        doc_B3: 0.014690    (MISSED)
        doc_B5: 0.014412    (MISSED)
        doc_L3: 0.014277    (MISSED)
        doc_L5: 0.014200    (MISSED)
        doc_L2: 0.014095    (MISSED)
        doc_L4: 0.014074    (MISSED)
        doc_T1: 0.011111    <- Start from here (DUP)
        doc_T2: 0.010938    (DUP)
        doc_T3: 0.010769    (DUP)
        doc_T4: 0.010606    (DUP)
        doc_T5: 0.010294    (DUP)
        """
        # Test with page size 5 for clean 3-page pagination (15 docs = 3 pages)
        page_size = 5
        all_paginated_hits = []

        # Collect all pages
        for page_num in range(3):  # 3 pages for 15 docs with page size 5
            offset = page_num * page_size

            page_res = tensor_search.search(
                search_method="HYBRID",
                hybrid_parameters=self.hp_rrf,
                config=self.config,
                index_name=self.index_unstructured.name,
                result_count=page_size,
                offset=offset,
                text=None
            )

            all_paginated_hits.extend(page_res["hits"])

        # TODO the result is quite broken, a lot of duplicates and missing docs. This can only be fixed by storing
        #  pagination state or over-fetching then trimming. This bad result is also related to the data set. In reality,
        #  we have a lot of docs matching both lexical and tensor, and with score modifiers, the result is much better.
        # Page 1
        self.assertEqual(['doc_B1', 'doc_B2', 'doc_T1', 'doc_T2', 'doc_T3'], [h['_id'] for h in all_paginated_hits[:5]])
        # Page 2
        self.assertEqual(['doc_T1', 'doc_T2', 'doc_T3', 'doc_T4', 'doc_T5'], [h['_id'] for h in all_paginated_hits[5:-5]])
        # Page 3
        self.assertEqual(['doc_T1', 'doc_T2', 'doc_T3', 'doc_T4', 'doc_T5'], [h['_id'] for h in all_paginated_hits[-5:]])

    @pytest.mark.skip_for_multinode
    def test_disjunction_rrf_pagination_with_unstable_tensor_ranking(self):
        """
        Test that disjunction+RRF pagination produces no duplicates between pages.
        Verifies ranking consistency and proper pagination behavior.

        Lexical search score:         Tensor search score:
        doc_B1: 1.5657032529354344    doc_B2: 0.8774912556666549
        doc_L1: 1.5310213335683778    doc_B1: 0.8624316166639774
        doc_L2: 1.4957020935493452    doc_T1: 0.8551202270206154
        doc_L3: 1.4403238705575117    doc_T2: 0.8550448320679322  <- will boost by 1, goes to the top in the first page result
        doc_L4: 1.3410214926606240    doc_T3: 0.8523972952471728
        doc_B3: 1.2495362288065737    doc_T4: 0.8500947166713401
        doc_L5: 1.1111902054958356    doc_B4: 0.8500164585022059
        doc_B5: 1.0981049722627436    doc_T5: 0.8471093151792289  <- will boost by 2, so it goes to the top, but only when we query page 2
        doc_B2: 0.9400916280884359    doc_B3: 0.8368122835299188
        doc_B4: 0.4681934225728979    doc_B5: 0.8334305830362274
                                      doc_L1: 0.8316111466678087
                                      doc_L5: 0.8288859450924600
                                      doc_L3: 0.8269666625377193
                                      doc_L4: 0.8263098149586303
                                      doc_L2: 0.8257278463225931

        Pagination with page size 5: with alpha=0.7, rrf_k=60
        tensor rrf score = alpha * (1.0 / (rank + k))
        lexical rrf score = (1-alpha) * (1.0 / (rank + k))

        ======================================================
        Page 1:
        Lexical search rrf score:     Tensor search rrf score:
        doc_B1: 0.004918              doc_T2: 0.011475        
        doc_L1: 0.004839              doc_B2: 0.011290        
        doc_L2: 0.004762              doc_B1: 0.011111        
        doc_L3: 0.004688              doc_T1: 0.010938        
        doc_L4: 0.004615              doc_T3: 0.010769        

        After fusion:
        doc_B1: 0.016029   <- In both, highest ranking
        doc_T2: 0.011475
        doc_B2: 0.011290
        doc_T1: 0.010938
        doc_T3: 0.010769
        doc_L1: 0.004839   <- Trimmed off from here
        doc_L2: 0.004762
        doc_L3: 0.004688
        doc_L4: 0.004615

        =======================================================
        Page 2:
        Lexical search rrf score:     Tensor search rrf score:
        doc_B1: 0.004918              doc_T5: 0.011475
        doc_L1: 0.004839              doc_T2: 0.011290
        doc_L2: 0.004762              doc_B2: 0.011111
        doc_L3: 0.004688              doc_B1: 0.010938
        doc_L4: 0.004615              doc_T1: 0.010769
        doc_B3: 0.004545              doc_T3: 0.010606
        doc_L5: 0.004478              doc_T4: 0.010448
        doc_B5: 0.004412              doc_B4: 0.010294
        doc_B2: 0.004348              doc_B3: 0.010145
        doc_B4: 0.004286              doc_B5: 0.01

        After fusion:
        doc_B1: 0.015856
        doc_B2: 0.015459
        doc_B3: 0.014690   (MISSED)
        doc_B4: 0.014580   (MISSED)
        doc_B5: 0.014412   (MISSED)
        doc_T5: 0.011475   <- Start from here
        doc_T2: 0.011290   (DUP)
        doc_T1: 0.010769   (DUP)
        doc_T3: 0.010606   (DUP)
        doc_T4: 0.010448
        doc_L1: 0.004839   <- Trimmed off from here
        doc_L2: 0.004762
        doc_L3: 0.004688
        doc_L4: 0.004615
        doc_L5: 0.004478

        =======================================================
        Page 3:
        Lexical search rrf score:     Tensor search rrf score:
        doc_B1: 0.004918              doc_T5: 0.011475
        doc_L1: 0.004839              doc_T2: 0.011290
        doc_L2: 0.004762              doc_B2: 0.011111
        doc_L3: 0.004688              doc_B1: 0.010938
        doc_L4: 0.004615              doc_T1: 0.010769
        doc_B3: 0.004545              doc_T3: 0.010606
        doc_L5: 0.004478              doc_T4: 0.010448
        doc_B5: 0.004412              doc_B4: 0.010294
        doc_B2: 0.004348              doc_B3: 0.010145
        doc_B4: 0.004286              doc_B5: 0.01
                                      doc_L1: 0.009859
                                      doc_L5: 0.009722
                                      doc_L3: 0.009589
                                      doc_L4: 0.009459
                                      doc_L2: 0.009333

        After fusion:
        doc_B1: 0.015856
        doc_B2: 0.015459
        doc_L1: 0.014698    (MISSED)
        doc_B3: 0.014690    (MISSED)
        doc_B4: 0.014580    (MISSED)
        doc_B5: 0.014412    (MISSED)
        doc_L3: 0.014277    (MISSED)
        doc_L5: 0.014200    (MISSED)
        doc_L2: 0.014095    (MISSED)
        doc_L4: 0.014074    (MISSED)
        doc_T5: 0.011475    <- Start from here (DUP)
        doc_T2: 0.011290    (DUP)
        doc_T1: 0.010769    (DUP)
        doc_T3: 0.010606    (DUP)
        doc_T4: 0.010448    (DUP)
        """
        # Test with page size 5 for clean 3-page pagination (15 docs = 3 pages)
        page_size = 5
        all_paginated_hits = []

        # Collect all pages
        for page_num in range(3):  # 3 pages for 15 docs with page size 5
            offset = page_num * page_size

            self.hp_rrf.scoreModifiersTensor = ScoreModifierLists(
                add_to_score=[
                    ScoreModifierOperator(field_name="tensor_score_boost_deep", weight=1),  # boost T2 by 1
                    ScoreModifierOperator(field_name="tensor_score_boost_data", weight=2),  # boost T5 by 2
                ]
            )

            page_res = tensor_search.search(
                search_method="HYBRID",
                hybrid_parameters=self.hp_rrf,
                config=self.config,
                index_name=self.index_unstructured.name,
                result_count=page_size,
                offset=offset,
                text=None
            )

            all_paginated_hits.extend(page_res["hits"])

        # Page 1
        self.assertEqual(['doc_B1', 'doc_T2', 'doc_B2', 'doc_T1', 'doc_T3'], [h['_id'] for h in all_paginated_hits[:5]])
        # Page 2
        self.assertEqual(['doc_T5', 'doc_T2', 'doc_T1', 'doc_T3', 'doc_T4'],
                         [h['_id'] for h in all_paginated_hits[5:-5]])
        # Page 3
        self.assertEqual(['doc_T5', 'doc_T2', 'doc_T1', 'doc_T3', 'doc_T4'],
                         [h['_id'] for h in all_paginated_hits[-5:]])

    @pytest.mark.skip_for_multinode
    def test_disjunction_rrf_pagination_with_global_modifiers_no_rerank_depth(self):
        """
        Test pagination with global score modifiers when rerankDepthGlobal is not set.
        All results get score modifiers applied.

        Lexical search score:         Tensor search score:
        doc_B1: rank 1  boost 11      doc_B2: rank 1   boost 12
        doc_L1: rank 2  boost 6       doc_B1: rank 2   boost 11
        doc_L2: rank 3  boost 7       doc_T1: rank 3   boost 1
        doc_L3: rank 4  boost 8       doc_T2: rank 4   boost 2
        doc_L4: rank 5  boost 9       doc_T3: rank 5   boost 3
        doc_B3: rank 6  boost 13      doc_T4: rank 6   boost 4
        doc_L5: rank 7  boost 10      doc_B4: rank 7   boost 14
        doc_B5: rank 8  boost 15      doc_T5: rank 8   boost 5
        doc_B2: rank 9  boost 12      doc_B3: rank 9   boost 13
        doc_B4: rank 10 boost 14      doc_B5: rank 10  boost 15
                                      doc_L1: rank 11  boost 6
                                      doc_L5: rank 12  boost 10
                                      doc_L3: rank 13  boost 8
                                      doc_L4: rank 14  boost 9
                                      doc_L2: rank 15  boost 7

        Pagination with page size 5: with alpha=0.7, rrf_k=60
        tensor rrf score = alpha * (1.0 / (rank + k))
        lexical rrf score = (1-alpha) * (1.0 / (rank + k))

        ======================================================
        Page 1:
        Lexical search rrf score:     Tensor search rrf score:
        doc_B1: 0.004918              doc_B2: 0.011475
        doc_L1: 0.004839              doc_B1: 0.011290
        doc_L2: 0.004762              doc_T1: 0.011111
        doc_L3: 0.004688              doc_T2: 0.010938
        doc_L4: 0.004615              doc_T3: 0.010769

        After fusion and apply score modifiers for all hits:
        doc_B2: 12.011475
        doc_B1: 11.016208
        doc_L4: 9.004615
        doc_L3: 8.004688
        doc_L2: 7.004762
        doc_L1: 6.004839   <- Trimmed off from here
        doc_T3: 3.010769
        doc_T2: 2.010938
        doc_T1: 1.011111

        =======================================================
        Page 2:
        Lexical search rrf score:     Tensor search rrf score:
        doc_B1: 0.004918              doc_B2: 0.011475
        doc_L1: 0.004839              doc_B1: 0.011290
        doc_L2: 0.004762              doc_T1: 0.011111
        doc_L3: 0.004688              doc_T2: 0.010938
        doc_L4: 0.004615              doc_T3: 0.010769
        doc_B3: 0.004545              doc_T4: 0.010606
        doc_L5: 0.004478              doc_B4: 0.010448
        doc_B5: 0.004412              doc_T5: 0.010294
        doc_B2: 0.004348              doc_B3: 0.010145
        doc_B4: 0.004286              doc_B5: 0.01

        After fusion:
        doc_B5: 15.014412  (MISSED)
        doc_B4: 14.014734  (MISSED)
        doc_B3: 13.014690  (MISSED)
        doc_B2: 12.015823
        doc_B1: 11.016208
        doc_L5: 10.004478  <- Start from here
        doc_L4: 9.004615   (DUP)
        doc_L3: 8.004688   (DUP)
        doc_L2: 7.004762   (DUP)
        doc_L1: 6.004839
        doc_T5: 5.010294   <- Trimmed off from here
        doc_T4: 4.010606
        doc_T3: 3.010769
        doc_T2: 2.010938
        doc_T1: 1.011111

        =======================================================
        Page 3:
        Lexical search rrf score:     Tensor search rrf score:
        doc_B1: 0.004918              doc_B2: 0.011475
        doc_L1: 0.004839              doc_B1: 0.011290
        doc_L2: 0.004762              doc_T1: 0.011111
        doc_L3: 0.004688              doc_T2: 0.010938
        doc_L4: 0.004615              doc_T3: 0.010769
        doc_B3: 0.004545              doc_T4: 0.010606
        doc_L5: 0.004478              doc_B4: 0.010448
        doc_B5: 0.004412              doc_T5: 0.010294
        doc_B2: 0.004348              doc_B3: 0.010145
        doc_B4: 0.004286              doc_B5: 0.01
                                      doc_L1: 0.009859
                                      doc_L5: 0.009722
                                      doc_L3: 0.009589
                                      doc_L4: 0.009459
                                      doc_L2: 0.009333

        After fusion:
        doc_B5: 15.014412  (MISSED)
        doc_B4: 14.014734  (MISSED)
        doc_B3: 13.014690  (MISSED)
        doc_B2: 12.015823
        doc_B1: 11.016208
        doc_L5: 10.014200
        doc_L4: 9.014074
        doc_L3: 8.014277
        doc_L2: 7.014095
        doc_L1: 6.014698
        doc_T5: 5.010294   <- Start from here
        doc_T4: 4.010606
        doc_T3: 3.010769
        doc_T2: 2.010938
        doc_T1: 1.011111
        """
        # Test with page size 5 for clean 3-page pagination
        page_size = 5
        all_paginated_hits = []

        # Collect all pages
        for page_num in range(3):  # 3 pages for 15 docs with page size 5
            offset = page_num * page_size

            page_res = tensor_search.search(
                search_method="HYBRID",
                hybrid_parameters=self.hp_rrf,
                config=self.config,
                index_name=self.index_unstructured.name,
                result_count=page_size,
                offset=offset,
                text=None,
                score_modifiers=ScoreModifierLists(
                    add_to_score=[
                        ScoreModifierOperator(field_name="score_boost", weight=1)
                    ]
                )
                # No rerankDepth - applies to all results
            )

            all_paginated_hits.extend(page_res["hits"])

        # Page 1
        self.assertEqual(['doc_B2', 'doc_B1', 'doc_L4', 'doc_L3', 'doc_L2'],
                         [h['_id'] for h in all_paginated_hits[:5]])
        # Page 2
        self.assertEqual(['doc_L5', 'doc_L4', 'doc_L3', 'doc_L2', 'doc_L1'],
                         [h['_id'] for h in all_paginated_hits[5:-5]])
        # Page 3
        self.assertEqual(['doc_T5', 'doc_T4', 'doc_T3', 'doc_T2', 'doc_T1'],
                         [h['_id'] for h in all_paginated_hits[-5:]])

    @pytest.mark.skip_for_multinode
    def test_disjunction_rrf_pagination_with_global_modifiers_with_rerank_depth(self):
        """
        Test pagination with rerankDepthGlobal values. The retrieved result up to offset+rerankDepthGlobal get global
        score modifiers applied

        Lexical search score:         Tensor search score:
        doc_B1: 1.5657032529354344    doc_B2: 0.8774912556666549
        doc_L1: 1.5310213335683778    doc_B1: 0.8624316166639774
        doc_L2: 1.4957020935493452    doc_T1: 0.8551202270206154
        doc_L3: 1.4403238705575117    doc_T2: 0.8550448320679322
        doc_L4: 1.3410214926606240    doc_T3: 0.8523972952471728
        doc_B3: 1.2495362288065737    doc_T4: 0.8500947166713401
        doc_L5: 1.1111902054958356    doc_B4: 0.8500164585022059
        doc_B5: 1.0981049722627436    doc_T5: 0.8471093151792289
        doc_B2: 0.9400916280884359    doc_B3: 0.8368122835299188
        doc_B4: 0.4681934225728979    doc_B5: 0.8334305830362274
                                      doc_L1: 0.8316111466678087
                                      doc_L5: 0.8288859450924600
                                      doc_L3: 0.8269666625377193
                                      doc_L4: 0.8263098149586303
                                      doc_L2: 0.8257278463225931

        Pagination with page size 5: with alpha=0.7, rrf_k=60
        tensor rrf score = alpha * (1.0 / (rank + k))
        lexical rrf score = (1-alpha) * (1.0 / (rank + k))

        ======================================================
        Page 1:
        Lexical search rrf score:     Tensor search rrf score:
        doc_B1: 0.004918              doc_B2: 0.011475
        doc_L1: 0.004839              doc_B1: 0.011290
        doc_L2: 0.004762              doc_T1: 0.011111
        doc_L3: 0.004688              doc_T2: 0.010938
        doc_L4: 0.004615              doc_T3: 0.010769

        After fusion:
        doc_B2: 12.011475
        doc_B1: 11.016208
        doc_L2: 7.004762
        doc_L1: 6.004839
        doc_T3: 3.010769
        doc_T2: 2.010938   <- Trimmed off from here
        doc_T1: 1.011111
        doc_L3: 0.004688   <- global reranking will not reach 8th element (rerankDepthGlobal = 7)
        doc_L4: 0.004615

        =======================================================
        Page 2:
        Lexical search rrf score:     Tensor search rrf score:
        doc_B1: 0.004918              doc_B2: 0.011475
        doc_L1: 0.004839              doc_B1: 0.011290
        doc_L2: 0.004762              doc_T1: 0.011111
        doc_L3: 0.004688              doc_T2: 0.010938
        doc_L4: 0.004615              doc_T3: 0.010769
        doc_B3: 0.004545              doc_T4: 0.010606
        doc_L5: 0.004478              doc_B4: 0.010448
        doc_B5: 0.004412              doc_T5: 0.010294
        doc_B2: 0.004348              doc_B3: 0.010145
        doc_B4: 0.004286              doc_B5: 0.01

        After fusion:
        doc_B5: 15.014412  (MISSED)
        doc_B4: 14.014734  (MISSED)
        doc_B3: 13.014690  (MISSED)
        doc_B2: 12.015823
        doc_B1: 11.016208
        doc_L2: 7.004762   <- Start from here (DUP)
        doc_L1: 6.004839   (DUP)
        doc_T5: 5.010294
        doc_T4: 4.010606
        doc_T3: 3.010769   (DUP)
        doc_T2: 2.010938   <- Trimmed off from here
        doc_T1: 1.011111
        doc_L3: 0.004688   <- global reranking will not reach 13th element (rerankDepthGlobal = 5 + 7)
        doc_L4: 0.004615
        doc_L5: 0.004478

        =======================================================
        Page 3:
        Lexical search rrf score:     Tensor search rrf score:
        doc_B1: 0.004918              doc_B2: 0.011475
        doc_L1: 0.004839              doc_B1: 0.011290
        doc_L2: 0.004762              doc_T1: 0.011111
        doc_L3: 0.004688              doc_T2: 0.010938
        doc_L4: 0.004615              doc_T3: 0.010769
        doc_B3: 0.004545              doc_T4: 0.010606
        doc_L5: 0.004478              doc_B4: 0.010448
        doc_B5: 0.004412              doc_T5: 0.010294
        doc_B2: 0.004348              doc_B3: 0.010145
        doc_B4: 0.004286              doc_B5: 0.01
                                      doc_L1: 0.009859
                                      doc_L5: 0.009722
                                      doc_L3: 0.009589
                                      doc_L4: 0.009459
                                      doc_L2: 0.009333

        After fusion: (global score modifiers applied to all docs)
        doc_B5: 15.014412  (MISSED)
        doc_B4: 14.014734  (MISSED)
        doc_B3: 13.014690  (MISSED)
        doc_B2: 12.015823
        doc_B1: 11.016208
        doc_L5: 10.014200  (MISSED)
        doc_L4: 9.014074   (MISSED)
        doc_L3: 8.014277   (MISSED)
        doc_L2: 7.014095
        doc_L1: 6.014698
        doc_T5: 5.010294   <- Start from here (DUP)
        doc_T4: 4.010606   (DUP)
        doc_T3: 3.010769   (DUP)
        doc_T2: 2.010938
        doc_T1: 1.011111
        """
        # Test with page size 5 for clean 3-page pagination
        page_size = 5
        all_paginated_hits = []

        # Collect all pages
        for page_num in range(3):  # 3 pages for 15 docs with page size 5
            offset = page_num * page_size

            page_res = tensor_search.search(
                search_method="HYBRID",
                hybrid_parameters=self.hp_rrf,
                config=self.config,
                index_name=self.index_unstructured.name,
                result_count=page_size,
                offset=offset,
                text=None,
                score_modifiers=ScoreModifierLists(
                    add_to_score=[
                        ScoreModifierOperator(field_name="score_boost", weight=1)
                    ]
                ),
                rerank_depth=7,  # rerank 7 items
            )

            all_paginated_hits.extend(page_res["hits"])

        # Page 1
        self.assertEqual(['doc_B2', 'doc_B1', 'doc_L2', 'doc_L1', 'doc_T3'],
                         [h['_id'] for h in all_paginated_hits[:5]])
        # Page 2
        self.assertEqual(['doc_L2', 'doc_L1', 'doc_T5', 'doc_T4', 'doc_T3'],
                         [h['_id'] for h in all_paginated_hits[5:-5]])
        # Page 3
        self.assertEqual(['doc_T5', 'doc_T4', 'doc_T3', 'doc_T2', 'doc_T1'],
                         [h['_id'] for h in all_paginated_hits[-5:]])

    @pytest.mark.skip_for_multinode
    def test_disjunction_rrf_pagination_with_global_modifiers_with_rerank_depth_small_than_limit(self):
        """
        Test pagination with rerankDepthGlobal values smaller than limit. The retrieved result up to
        offset+rerankDepthGlobal get global score modifiers applied

        Lexical search score:         Tensor search score:
        doc_B1: 1.5657032529354344    doc_B2: 0.8774912556666549
        doc_L1: 1.5310213335683778    doc_B1: 0.8624316166639774
        doc_L2: 1.4957020935493452    doc_T1: 0.8551202270206154
        doc_L3: 1.4403238705575117    doc_T2: 0.8550448320679322
        doc_L4: 1.3410214926606240    doc_T3: 0.8523972952471728
        doc_B3: 1.2495362288065737    doc_T4: 0.8500947166713401
        doc_L5: 1.1111902054958356    doc_B4: 0.8500164585022059
        doc_B5: 1.0981049722627436    doc_T5: 0.8471093151792289
        doc_B2: 0.9400916280884359    doc_B3: 0.8368122835299188
        doc_B4: 0.4681934225728979    doc_B5: 0.8334305830362274
                                      doc_L1: 0.8316111466678087
                                      doc_L5: 0.8288859450924600
                                      doc_L3: 0.8269666625377193
                                      doc_L4: 0.8263098149586303
                                      doc_L2: 0.8257278463225931

        Pagination with page size 5: with alpha=0.7, rrf_k=60
        tensor rrf score = alpha * (1.0 / (rank + k))
        lexical rrf score = (1-alpha) * (1.0 / (rank + k))

        ======================================================
        Page 1:
        Lexical search rrf score:     Tensor search rrf score:
        doc_B1: 0.004918              doc_B2: 0.011475
        doc_L1: 0.004839              doc_B1: 0.011290
        doc_L2: 0.004762              doc_T1: 0.011111
        doc_L3: 0.004688              doc_T2: 0.010938
        doc_L4: 0.004615              doc_T3: 0.010769

        After fusion and applying global score modifiers
        doc_B2: 12.011475
        doc_B1: 11.016208
        doc_T1: 1.011111
        doc_T2: 0.010938  <- global reranking will not reach 4th element (rerankDepthGlobal = 3)
        doc_T3: 0.010769
        doc_L1: 0.004839  <- Trimmed off from here
        doc_L2: 0.004762
        doc_L3: 0.004688
        doc_L4: 0.004615

        =======================================================
        Page 2:
        Lexical search rrf score:     Tensor search rrf score:
        doc_B1: 0.004918              doc_B2: 0.011475
        doc_L1: 0.004839              doc_B1: 0.011290
        doc_L2: 0.004762              doc_T1: 0.011111
        doc_L3: 0.004688              doc_T2: 0.010938
        doc_L4: 0.004615              doc_T3: 0.010769
        doc_B3: 0.004545              doc_T4: 0.010606
        doc_L5: 0.004478              doc_B4: 0.010448
        doc_B5: 0.004412              doc_T5: 0.010294
        doc_B2: 0.004348              doc_B3: 0.010145
        doc_B4: 0.004286              doc_B5: 0.01

        After fusion:
        doc_B5: 15.014412  (MISSED)
        doc_B4: 14.014734  (MISSED)
        doc_B3: 13.014690  (MISSED)
        doc_B2: 12.015823
        doc_B1: 11.016208
        doc_T3: 3.010769   <- Start from here (DUP)
        doc_T2: 2.010938   (DUP)
        doc_T1: 1.011111   (DUP)
        doc_T4: 0.010606   <- global reranking will not reach 9th element (rerankDepthGlobal = 5 + 3)
        doc_T5: 0.010294
        doc_L1: 0.004839   <- Trimmed off from here
        doc_L2: 0.004762
        doc_L3: 0.004688
        doc_L4: 0.004615
        doc_L5: 0.004478

        =======================================================
        Page 3:
        Lexical search rrf score:     Tensor search rrf score:
        doc_B1: 0.004918              doc_B2: 0.011475
        doc_L1: 0.004839              doc_B1: 0.011290
        doc_L2: 0.004762              doc_T1: 0.011111
        doc_L3: 0.004688              doc_T2: 0.010938
        doc_L4: 0.004615              doc_T3: 0.010769
        doc_B3: 0.004545              doc_T4: 0.010606
        doc_L5: 0.004478              doc_B4: 0.010448
        doc_B5: 0.004412              doc_T5: 0.010294
        doc_B2: 0.004348              doc_B3: 0.010145
        doc_B4: 0.004286              doc_B5: 0.01
                                      doc_L1: 0.009859
                                      doc_L5: 0.009722
                                      doc_L3: 0.009589
                                      doc_L4: 0.009459
                                      doc_L2: 0.009333

        After fusion: (global score modifiers applied to all docs)
        doc_B5: 15.014412  (MISSED)
        doc_B4: 14.014734  (MISSED)
        doc_B3: 13.014690  (MISSED)
        doc_B2: 12.015823
        doc_B1: 11.016208
        doc_L5: 10.014200  (MISSED)
        doc_L4: 9.014074   (MISSED)
        doc_L3: 8.014277   (MISSED)
        doc_L2: 7.014095   (MISSED)
        doc_L1: 6.014698   (MISSED)
        doc_T3: 3.010769   (DUP)
        doc_T2: 2.010938   (DUP)
        doc_T1: 1.011111   (DUP)
        doc_T4: 0.010606   <- global reranking will not reach 14th element (rerankDepthGlobal = 10 + 3)  (DUP)
        doc_T5: 0.010294   (DUP)
        """
        # Test with page size 5 for clean 3-page pagination
        page_size = 5
        all_paginated_hits = []

        # Collect all pages
        for page_num in range(3):  # 3 pages for 15 docs with page size 5
            offset = page_num * page_size

            page_res = tensor_search.search(
                search_method="HYBRID",
                hybrid_parameters=self.hp_rrf,
                config=self.config,
                index_name=self.index_unstructured.name,
                result_count=page_size,
                offset=offset,
                text=None,
                score_modifiers=ScoreModifierLists(
                    add_to_score=[
                        ScoreModifierOperator(field_name="score_boost", weight=1)
                    ]
                ),
                rerank_depth=3,  # rerank only 3 items
            )

            all_paginated_hits.extend(page_res["hits"])

        # Page 1
        self.assertEqual(['doc_B2', 'doc_B1', 'doc_T1', 'doc_T2', 'doc_T3'],
                         [h['_id'] for h in all_paginated_hits[:5]])
        # Page 2
        self.assertEqual(['doc_T3', 'doc_T2', 'doc_T1', 'doc_T4', 'doc_T5'],
                         [h['_id'] for h in all_paginated_hits[5:-5]])
        # Page 3
        self.assertEqual(['doc_T3', 'doc_T2', 'doc_T1', 'doc_T4', 'doc_T5'],
                         [h['_id'] for h in all_paginated_hits[-5:]])
        