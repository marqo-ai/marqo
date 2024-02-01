from unittest import mock

import pydantic
import pytest

from marqo import exceptions as base_exceptions
from marqo.api import exceptions as api_exceptions
from marqo.api.exceptions import InvalidArgError
from marqo.core import exceptions as core_exceptions
from marqo.tensor_search.api import marqo_base_exception_handler
from marqo.vespa import exceptions as vespa_exceptions
from tests.marqo_test import MarqoTestCase


class BaseMarqoModel(pydantic.BaseModel):
    class Config:
        extra: str = "forbid"

    pass


class TestBaseExceptionHandler(MarqoTestCase):
    """
    Ensure that calling the base exception handler with a base/core exception calls the API exception handler
    with the correct converted api exception.
    """

    def setUp(self) -> None:
        super().setUp()
        self.normal_request = mock.MagicMock()
        # Create a request instance with headers and a JSON body
        headers = {'Content-Type': 'application/json'}
        body = b'{"name": "Alice", "age": 30}'

        # Create a mock request object
        self.normal_request = mock.MagicMock()
        self.normal_request.headers = headers
        self.normal_request.content = body
        self.generic_error_message = "This is an error!"
        self.mock_api_exception_handler = mock.MagicMock()

    @mock.patch("marqo.tensor_search.api.marqo_api_exception_handler")
    def test_base_exception_handler_base_errors(self, mock_api_exception_handler):
        with self.subTest("Base error: Internal Error"):
            marqo_base_exception_handler(self.normal_request, base_exceptions.InternalError(self.generic_error_message))
            assert isinstance(mock_api_exception_handler.call_args_list[-1][0][1], api_exceptions.InternalError)  # 500

        with self.subTest("Base error: Invalid Argument Error"):
            marqo_base_exception_handler(self.normal_request,
                                         base_exceptions.InvalidArgumentError(self.generic_error_message))
            assert isinstance(mock_api_exception_handler.call_args_list[-1][0][1],
                              api_exceptions.InvalidArgError)  # 400

    @mock.patch("marqo.tensor_search.api.marqo_api_exception_handler")
    def test_base_exception_handler_core_errors(self, mock_api_exception_handler):
        with self.subTest("Core error: Index Exists Error"):
            marqo_base_exception_handler(self.normal_request,
                                         core_exceptions.IndexExistsError(self.generic_error_message))
            assert isinstance(mock_api_exception_handler.call_args_list[-1][0][1],
                              api_exceptions.IndexAlreadyExistsError)  # 409

        with self.subTest("Core error: Index Not Found Error"):
            marqo_base_exception_handler(self.normal_request,
                                         core_exceptions.IndexNotFoundError(self.generic_error_message))
            assert isinstance(mock_api_exception_handler.call_args_list[-1][0][1],
                              api_exceptions.IndexNotFoundError)  # 404

        with self.subTest("Core error: Parsing Error"):
            marqo_base_exception_handler(self.normal_request, core_exceptions.ParsingError(self.generic_error_message))
            assert isinstance(mock_api_exception_handler.call_args_list[-1][0][1], api_exceptions.MarqoWebError)  # 500

        with self.subTest("Core error: Vespa Document Parsing Error"):
            marqo_base_exception_handler(self.normal_request,
                                         core_exceptions.VespaDocumentParsingError(self.generic_error_message))
            assert isinstance(mock_api_exception_handler.call_args_list[-1][0][1],
                              api_exceptions.BackendDataParsingError)

        with self.subTest("Core error: Marqo Document Parsing Error"):
            marqo_base_exception_handler(self.normal_request,
                                         core_exceptions.MarqoDocumentParsingError(self.generic_error_message))
            assert isinstance(mock_api_exception_handler.call_args_list[-1][0][1], api_exceptions.InvalidArgError)

        with self.subTest("Core error: Invalid Data Type Error"):
            marqo_base_exception_handler(self.normal_request,
                                         core_exceptions.InvalidDataTypeError(self.generic_error_message))
            assert isinstance(mock_api_exception_handler.call_args_list[-1][0][1], api_exceptions.InvalidArgError)

        with self.subTest("Core error: Invalid Field Name Error"):
            marqo_base_exception_handler(self.normal_request,
                                         core_exceptions.InvalidFieldNameError(self.generic_error_message))
            assert isinstance(mock_api_exception_handler.call_args_list[-1][0][1], api_exceptions.InvalidFieldNameError)

        with self.subTest("Core error: Filter String Parsing Error"):
            marqo_base_exception_handler(self.normal_request,
                                         core_exceptions.FilterStringParsingError(self.generic_error_message))
            assert isinstance(mock_api_exception_handler.call_args_list[-1][0][1], api_exceptions.InvalidArgError)

    @mock.patch("marqo.tensor_search.api.marqo_api_exception_handler")
    def test_base_exception_handler_vespa_errors(self, mock_api_exception_handler):
        with self.subTest("Vespa error: Vespa Error"):
            marqo_base_exception_handler(self.normal_request, vespa_exceptions.VespaError(self.generic_error_message))
            assert isinstance(mock_api_exception_handler.call_args_list[-1][0][1], api_exceptions.MarqoWebError)
