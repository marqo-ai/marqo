from marqo.s2_inference.processing.text import split_text
import unittest
import copy


class TestSplitText(unittest.TestCase):

    def setUp(self) -> None:
        self.empty_list = []
        self.none_string = None
        self.empty_string1 = ""
        self.empty_string2 = ''

        self.split_by_valid = ['character', 'word', 'sentence']

        self.wrong_split_by = 'sasa'

    def test_split_split_by(self):
        
        try:
            result = split_text(self.empty_string1, split_by=self.wrong_split_by)
        except KeyError as e:
            assert self.wrong_split_by in str(e)

        for split_by in self.split_by_valid:
            result = split_text(self.empty_string1, split_by=split_by)

    def test_split_text_empty(self):
        
        for split_by in self.split_by_valid:

            result = split_text(self.empty_string1, split_by=split_by)
            assert result == [' ']
            result = split_text(self.empty_string2, split_by=split_by)
            assert result == [' ']
            result = split_text(self.empty_list, split_by=split_by)
            assert result == [' ']
            result = split_text(self.none_string, split_by=split_by)
            assert result == [' ']

    def test_split_text_single_character(self):
        text = "a"
        for split_by in self.split_by_valid:
            result = split_text(text, split_by=split_by)
            assert result == [text]

    def test_split_text_whitespace(self):
        ws = [
            " ",
            "\r",
            "   ",
            "\r\t",
            "\r  \t"
        ]

        for text in ws:
            for split_by in self.split_by_valid:
                result = split_text(text, split_by=split_by)
                assert result == [" "]

    def test_split_text_single_word(self):
        
        text = 'short'
        result = split_text(text, split_by='character', split_length=4, split_overlap=1)
        assert result == ['shor', 'rt']

        result = split_text(text, split_by='character', split_length=4, split_overlap=2)
        assert result == ['shor', 'ort']

        result = split_text(text, split_by='character', split_length=4, split_overlap=3)
        assert result == ['shor', 'hort']

        result = split_text(text, split_by='character', split_length=1, split_overlap=0)
        assert result == list(text)

        result = split_text(text, split_by='word', split_length=4, split_overlap=1)
        assert result == [text]

        result = split_text(text, split_by='sentence', split_length=4, split_overlap=1)
        assert result == [text]

