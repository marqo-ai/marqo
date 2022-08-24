import requests

from marqo.client import Client
from marqo.errors import MarqoApiError
import unittest
import pprint
from tests.marqo_test import MarqoTestCase
import tempfile
from PIL import Image
import numpy as np
import os
from marqo.processing import text as text_processor

class TestSentenceChunking(MarqoTestCase):
    """Test for sentence chunking

    Assumptions:
        - Local OpenSearch (not S2Search)
    """
    def setUp(self) -> None:
        client_0 = Client(**self.client_settings)
        
        self.index_name = 'sentence-chunk-test'

        try:
            client_0.delete_index(self.index_name)
        except MarqoApiError as s:
            pass

    def test_sentence_no_chunking(self):

        client = Client(**self.client_settings)
        
        try:
            client.delete_index(self.index_name)
        except MarqoApiError as s:
            pass

        settings = {
            "sentences_per_chunk":int(1e3),  
            "sentence_overlap":0 
            }
        
        client.create_index(self.index_name, **settings)


        document1 = {'_id': '1', # '_id' can be provided but is not required
            'attributes': 'hello. how are you. another one.',
            'description': 'the image chunking. can (optionally) chunk. the image into sub-patches (aking to segmenting text). by using either. a learned model. or simple box generation and cropping.',
            'misc':'sasasasaifjfnonfqeno asadsdljknjdfln'}

        client.index(self.index_name).add_documents([document1])

        # test the search works
        results = client.index(self.index_name).search('a')
        print(results)
        assert results['hits'][0]['attributes'] == document1['attributes']

        assert results['hits'][0]['description'] == document1['description']

        assert results['hits'][0]['misc'] == document1['misc']

    def test_sentence_chunking_no_overlap(self):

        client = Client(**self.client_settings)
        
        try:
            client.delete_index(self.index_name)
        except MarqoApiError as s:
            pass

        settings = {
            "sentences_per_chunk":2,  
            "sentence_overlap":0 
            }
        
        client.create_index(self.index_name, **settings)


        document1 = {'_id': '1', # '_id' can be provided but is not required
            'attributes': 'hello. how are you. another one.',
            'description': 'the image chunking. can (optionally) chunk. the image into sub-patches (aking to segmenting text). by using either. a learned model. or simple box generation and cropping.',
            'misc':'sasasasaifjfnonfqeno asadsdljknjdfln'}

        client.index(self.index_name).add_documents([document1])

        # search with a term we know is an exact chunk and will then show in the highlights
        search_term = 'hello. how are you.'
        results = client.index(self.index_name).search(search_term)
        print(results)
        assert results['hits'][0]['_highlights']['attributes'] == search_term

        # search with a term we know is an exact chunk and will then show in the highlights
        search_term = 'the image into sub-patches (aking to segmenting text). by using either.'
        results = client.index(self.index_name).search(search_term)
        print(results)
        assert results['hits'][0]['_highlights']['description'] == search_term

        # search with a term we know is an exact chunk and will then show in the highlights
        search_term = 'sasasasaifjfnonfqeno asadsdljknjdfln'
        results = client.index(self.index_name).search(search_term)
        print(results)
        assert results['hits'][0]['_highlights']['misc'] == search_term

        # search with a term we know is part of a sub-chunk and verify it is overlapping in the correct sentence
        search_term = 'can (optionally) chunk.'
        results = client.index(self.index_name).search(search_term)
        print(results)
        assert results['hits'][0]['_highlights']['description'] == 'the image chunking. can (optionally) chunk.'

    def test_sentence_chunking_overlap(self):

        client = Client(**self.client_settings)
        
        try:
            client.delete_index(self.index_name)
        except MarqoApiError as s:
            pass

        settings = {
            "sentences_per_chunk":2,  
            "sentence_overlap":1
            }
        
        client.create_index(self.index_name, **settings)


        document1 = {'_id': '1', # '_id' can be provided but is not required
            'attributes': 'hello. how are you. another one.',
            'description': 'the image chunking. can (optionally) chunk. the image into sub-patches (aking to segmenting text). by using either. a learned model. or simple box generation and cropping.',
            'misc':'sasasasaifjfnonfqeno asadsdljknjdfln'}

        client.index(self.index_name).add_documents([document1])

        # search with a term we know is an exact chunk and will then show in the highlights
        search_term = 'hello. how are you.'
        results = client.index(self.index_name).search(search_term)
        print(results)
        assert results['hits'][0]['_highlights']['attributes'] == search_term

        # search with a term we know is an exact chunk and will then show in the highlights
        search_term = 'the image into sub-patches (aking to segmenting text). by using either.'
        results = client.index(self.index_name).search(search_term)
        print(results)
        assert results['hits'][0]['_highlights']['description'] == search_term

        # search with a term we know is an exact chunk and will then show in the highlights
        search_term = 'sasasasaifjfnonfqeno asadsdljknjdfln'
        results = client.index(self.index_name).search(search_term)
        print(results)
        assert results['hits'][0]['_highlights']['misc'] == search_term

        # search with a term we know is part of a sub-chunk and verify it is overlapping in the correct sentence
        search_term = 'can (optionally) chunk.'
        results = client.index(self.index_name).search(search_term)
        print(results)
        assert results['hits'][0]['_highlights']['description'] == 'the image chunking. can (optionally) chunk.'

        # check overlap
        search_term = "can (optionally) chunk. the image into sub-patches (aking to segmenting text)."
        results = client.index(self.index_name).search(search_term)
        print(results)
        assert results['hits'][0]['_highlights']['description'] == search_term

        search_term = "the image into sub-patches (aking to segmenting text). by using either."
        results = client.index(self.index_name).search(search_term)
        print(results)
        assert results['hits'][0]['_highlights']['description'] == search_term
