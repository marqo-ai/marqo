import unittest
import os
from marqo.s2_inference.enrichment.vqa_utils import _process_kwargs_vqa

class TestEncoding(unittest.TestCase):

    def setUp(self) -> None:

        pass

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