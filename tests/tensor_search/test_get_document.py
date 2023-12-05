import functools
import pprint
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search import enums
from marqo.api.exceptions import IndexNotFoundError, InvalidDocumentIdError
from marqo.tensor_search import tensor_search
from tests.marqo_test import MarqoTestCase


class TestGetDocument(MarqoTestCase):

    def setUp(self) -> None:
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"
        self.index_name_2 = "my-test-index-2"
        self._delete_testing_indices()

    def _delete_testing_indices(self):
        for ix in [self.index_name_1, self.index_name_2]:
            try:
                tensor_search.delete_index(config=self.config, index_name=ix)
            except IndexNotFoundError as s:
                pass

    def test_get_document(self):
        """Also ensures that the _id is returned"""
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        tensor_search.add_documents(
            config=self.config, add_docs_params=AddDocsParams(index_name=self.index_name_1, docs=[
                {
                    "_id": "123",
                    "title 1": "content 1",
                    "desc 2": "content 2. blah blah blah"
                }], auto_refresh=True, device="cpu")
        )
        assert tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="123") == {
                "_id": "123",
                "title 1": "content 1",
                "desc 2": "content 2. blah blah blah"
            }

    def test_get_document_non_existent_index(self):
        try:
            a = tensor_search.get_document_by_id(
                config=self.config, index_name=self.index_name_1,
                document_id="123")
            raise AssertionError
        except IndexNotFoundError as e:
            pass

    def test_get_document_empty_str(self):
        try:
            a = tensor_search.get_document_by_id(
                config=self.config, index_name=self.index_name_1,
                document_id="")
            print(a)
            raise AssertionError
        except InvalidDocumentIdError as e:
            pass

    def test_get_document_bad_types(self):
        for ar in [12.2, 1, [], {}, None]:
            try:
                a = tensor_search.get_document_by_id(
                    config=self.config, index_name=self.index_name_1,
                    document_id=ar)
                raise AssertionError
            except InvalidDocumentIdError as e:
                pass

    def test_get_document_vectors_format(self):
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1)
        keys = ("title 1", "desc 2")
        vals = ("content 1", "content 2. blah blah blah")
        tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, docs=[{"_id": "123", **dict(zip(keys, vals))}],
            auto_refresh=True, device="cpu")
        )
        res = tensor_search.get_document_by_id(
            config=self.config, index_name=self.index_name_1,
            document_id="123", show_vectors=True)
        assert enums.TensorField.tensor_facets in res
        assert len(res[enums.TensorField.tensor_facets]) == 2
        assert set(keys) == functools.reduce(
            lambda x, y: x.union(y),
            [set(facet.keys()) for facet in res[enums.TensorField.tensor_facets]]
        ) - {enums.TensorField.embedding}
        for facet in res[enums.TensorField.tensor_facets]:
            assert len(facet) == 2
            if keys[0] in facet:
                assert facet[keys[0]] == vals[0]
            if keys[1] in facet:
                assert facet[keys[1]] == vals[1]
            assert enums.TensorField.embedding in facet


