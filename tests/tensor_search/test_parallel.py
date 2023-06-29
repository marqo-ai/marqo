from marqo.errors import IndexNotFoundError
from marqo.tensor_search import parallel
import torch
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
import os
from unittest import mock
from marqo.errors import InternalError

class TestAddDocumentsPara(MarqoTestCase):
    """
    This test generates SSL warnings when running against a local Marqo because
    parallel.py turns on logging.
    """

    pass
        