import time
import math
import pprint
from unittest import mock
from marqo.tensor_search.enums import TensorField, SearchMethod, EnvVars
from marqo.errors import (
    MarqoApiError, MarqoError, IndexNotFoundError, InvalidArgError,
    InvalidFieldNameError, IllegalRequestedDocCount
)
from marqo.tensor_search import tensor_search, constants, index_meta_cache
import copy
from tests.marqo_test import MarqoTestCase
import requests
import random

class TestThrottling(MarqoTestCase):

    def setUp(self) -> None:
        self.endpoint = self.authorized_url
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "my-test-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def tearDown(self) -> None:
        self.index_name_1 = "my-test-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass
    
    
    
    def test_throttle_decrement_on_400_error(self):

        mock_redis_driver = MagicMock()
        db = MagicMock()
        
        def increment_counter(*args, **kwargs):
            db.counter += 1
        
        def decrement_counter(*args, **kwargs):
            db.counter -= 1

        db.evalsha.side_effect = increment_counter
        db.zrem.side_effect = decrement_counter
        db.counter = 0
        mock_redis_driver.get_db.return_value = db

        @mock.patch("marqo.connections.redis_driver", mock_redis_driver)
        def run():
            try:
                # Index with bad request
                tensor_search.add_documents(config=self.config, index_name=self.index_name_1, 
                    docs=[{"blah": "blah"}], batch_size=-1, auto_refresh=True)
                raise AssertionError("Negative batch size should fail!")
            
            except:
                # After 400 error, wait a second or so, check that count from redis driver is the same
                assert db.counter == 0

        assert run()
    
    def test_throttle_decrement_on_500_error(self):
        # Empty redis
        # Mock marqo-os doesn't exist
        # Try to index something
        # Note the thread name
        # After 500 error, check that thread is deleted.
        pass

        
        