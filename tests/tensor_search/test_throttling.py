import time
import math
import pprint
from unittest import mock
from marqo.tensor_search.enums import TensorField, SearchMethod, EnvVars
from marqo.api.exceptions import (
    IndexNotFoundError, InvalidArgError,
    InvalidFieldNameError, IllegalRequestedDocCount
)
from marqo.tensor_search import tensor_search, constants, index_meta_cache
from marqo.tensor_search.throttling.redis_throttle import throttle

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
    
    
    # TODO: Fix this unit test and add more.
    """
    def test_throttle_decrement_on_error(self):
        
        mock_redis_driver = mock.MagicMock()
        db = mock.MagicMock()

        mock_redis_driver.get_db.return_value = db
        
        def increment_counter(*args, **kwargs):
            db.counter += 1
            print(f"inc: db counter is {db.counter}")
        
        def decrement_counter(*args, **kwargs):
            db.counter -= 1
            print(f"dec: db counter is {db.counter}")

        def tester():
            print(f"counter: {db.counter}")
        db.evalsha.side_effect = increment_counter
        db.zrem.side_effect = decrement_counter
        db.counter = 0

        @throttle("INDEX")
        def func_that_dies():
            raise Exception("Some Marqo error occured.")
        
        @mock.patch("marqo.connections.redis_driver", mock_redis_driver)
        def run():
            try:
                func_that_dies()

            except Exception as e:
                # After error, wait a second or so, check that count from redis driver is the same
                print(e)
                assert db.counter == 0
                return True

        assert run()
        """