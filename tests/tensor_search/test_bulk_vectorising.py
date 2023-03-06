from marqo.tensor_search import bulk_vectorise
from tests.marqo_test import MarqoTestCase


class TestBulkVectorise(MarqoTestCase):

    def test_Content4Vectorising_dict_equality(self):
        """do two equal Content4Vectorising objects take up the same
        dictionary entry?
        """
        c1 = bulk_vectorise.Content4Vectorising(string_representation="hallo")
        c2 = bulk_vectorise.Content4Vectorising(string_representation="hallo")
        assert c1 is not c2
        assert c1 == c2
        my_dict = {c1: "wow"}
        assert my_dict[c2] == "wow"

    def test_Content4Vectorising_eq_separate_classes(self):
        class FakeContent4Vec:
            def __init__(self, string_representation: str):
                self.string_representation = string_representation
                self.content_to_vectorise = string_representation

            def __hash__(self):
                return hash(self.string_representation)

        c1 = bulk_vectorise.Content4Vectorising(string_representation="hallo")
        c2 = FakeContent4Vec(string_representation="hallo")
        assert c1 is not c2
        assert c1 != c2
        assert hash(c1) == hash(c2)
        my_dict = {c1: "wow"}
        try:
            my_dict[c2] == "wow"
            raise AssertionError
        except KeyError:
            pass
