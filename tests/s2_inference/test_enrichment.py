import unittest
import os
from marqo.s2_inference.enrichment.vqa_utils import (
    _process_kwargs_vqa,
    _check_output_type,
    convert_to_bool,
    get_allowed_vqa_tasks,
    VQA
)

from marqo.s2_inference.s2_inference import generate


class TestEncoding(unittest.TestCase):

    def setUp(self) -> None:

        pass

    def test_check_vqa_output_type(self):
        
        outputs = [
            [],
            [1],
            ['a'],
            [0],
            [.1],
            [1,'a',.1],
            ['a', 'asasas']
        ]
        for output in outputs:
            assert _check_output_type(output)

        outputs = [
            'a',
            1,
            None, 
            (1)
        ]
        for output in outputs:
            assert not _check_output_type(output)

    def test_convert_to_bool(self):
        
        answers = ['yes', True, 'True', 'true', 1, '1', 'y', 't']
        for answer in answers:
            assert convert_to_bool(answer)

        answers = ['no', False, 'False', 'false', 0, '0', 'n', 'f']
        for answer in answers:
            assert not convert_to_bool(answer)

    def test_allowed_vqa_tasks(self):
        allowed_tasks = get_allowed_vqa_tasks()
        assert "attribute-extraction" in allowed_tasks 
        assert "question-answer" in allowed_tasks 

    def test_process_kwargs_vqa(self):

        # attributes list
        kwargs_dict = {
                "attributes": ["Bathroom, Bedroom, Study, Yard"],
                "image_field": "https://s3.image.png"
            }

        expected_output = [('https://s3.image.png', 'Bathroom'),
                        ('https://s3.image.png', 'Bedroom'),
                        ('https://s3.image.png', 'Study'),
                        ('https://s3.image.png', 'Yard')]

        image_query_pairs = _process_kwargs_vqa(kwargs_dict=kwargs_dict)

        assert image_query_pairs == expected_output

        # attributes
        kwargs_dict = {
                "attributes": "Bathroom, Bedroom, Study, Yard",
                "image_field": "https://s3.image.png"
            }

        expected_output = [('https://s3.image.png', 'Bathroom'),
                        ('https://s3.image.png', 'Bedroom'),
                        ('https://s3.image.png', 'Study'),
                        ('https://s3.image.png', 'Yard')]

        image_query_pairs = _process_kwargs_vqa(kwargs_dict=kwargs_dict)

        assert image_query_pairs == expected_output

        # query
        kwargs_dict = {
                "query": ["how many lights are in the room?"],
                "image_field": "https://s3.image.png"
            }

        expected_output =  [('https://s3.image.png', 'how many lights are in the room?')]

        image_query_pairs = _process_kwargs_vqa(kwargs_dict=kwargs_dict)

        assert image_query_pairs == expected_output
        
        # query as a string only
        kwargs_dict = {
                "query": "how many lights are in the room?",
                "image_field": "https://s3.image.png"
            }

        expected_output =  [('https://s3.image.png', 'how many lights are in the room?')]

        image_query_pairs = _process_kwargs_vqa(kwargs_dict=kwargs_dict)

        assert image_query_pairs == expected_output


        # queries
        kwargs_dict = {
                "query": ["how many lights are in the room?", "how many chairs in the room?"],
                "image_field": "https://s3.image.png"
            }

        expected_output = [('https://s3.image.png', 'how many lights are in the room?'),
                 ('https://s3.image.png', 'how many chairs in the room?')]        
        
        image_query_pairs = _process_kwargs_vqa(kwargs_dict=kwargs_dict)

        assert image_query_pairs == expected_output

        # TODO add tests for erros

    def test_vqa_model(self):
        
        model = VQA()
        model.load()
        
        task = "attribute-extraction"
        # output from _parse_kwargs in enrichment.py
        parsed_kwargs = {
                "attributes": ["hippo, water, car, truck"],
                "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
            }

        model.predict(task, [], **parsed_kwargs)

        assert model.answers == [True, True, False, False]

        task = "question-answer"

        # output from _parse_kwargs in enrichment.py
        parsed_kwargs = {
                "query": "how many hippos are in the picture?",
                "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
            }

        model.predict(task, [], **parsed_kwargs)

        assert model.answers == ['one']
    
    def test_generate(self):
        """
                generate(
        "attribute-extraction", 
        [ ],
        {
                    "attributes": ["Bathroom, Bedroom, Study, Yard"]
                    "image_field": "https://s3.image.png"
                 }
        )
        """
        task = "attribute-extraction"
        # output from _parse_kwargs in enrichment.py
        parsed_kwargs = {
                "attributes": ["hippo, water, car, truck"],
                "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
            }

        results = generate(task, 'cpu', [], **parsed_kwargs)

        assert results == [True, True, False, False]


        task = "question-answer"

        # output from _parse_kwargs in enrichment.py
        parsed_kwargs = {
                "query": "how many hippos are in the picture?",
                "image_field": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
            }

        results = generate(task, 'cpu', [], **parsed_kwargs)

        assert results == ['one']
