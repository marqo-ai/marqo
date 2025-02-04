from unittest import mock
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError

from marqo import exceptions as base_exceptions
from marqo.api import exceptions as api_exceptions
from marqo.api.exceptions import UnprocessableEntityError
from marqo.core import exceptions as core_exceptions
from marqo.tensor_search.api import api_validation_exception_handler
from marqo.tensor_search.api import marqo_base_exception_handler
from marqo.vespa import exceptions as vespa_exceptions
from integ_tests.marqo_test import MarqoTestCase


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

    @mock.patch("marqo.tensor_search.api.marqo_api_exception_handler")
    def test_base_exception_handler_unhandled_error(self, mock_api_exception_handler):
        # Ensure that an unhandled error is converted to a MarqoWebError and the original message is not propagated
        marqo_base_exception_handler(self.normal_request, base_exceptions.MarqoError("This should not be propagated."))
        converted_error = mock_api_exception_handler.call_args_list[-1][0][1]
        
        self.assertIsInstance(converted_error, api_exceptions.MarqoWebError)
        self.assertNotIn("This should not be propagated.", converted_error.message)
        self.assertIn("unexpected internal error", converted_error.message)

    @pytest.mark.asyncio
    async def test_validation_exception_handler_CorrectResponseBody(self):
        """Ensure that the validation exception handler returns the correct response body when 422 is raised."""
        # Create a mock request
        mock_request = MagicMock(spec=Request)
        # Manually create a RequestValidationError
        errors = [
            {
                "loc": ("body", "name"),
                "msg": "ensure this value has at most 10 characters",
                "type": "value_error.any_str.max_length",
                "ctx": {"limit_value": 10}
            }
        ]
        validation_error = RequestValidationError(errors)

        response = await api_validation_exception_handler(mock_request, validation_error)
        self.assertEqual(422, response.status_code)
        self.assertEqual(
            {
                "detail": jsonable_encoder(validation_error.errors()),
                "code": UnprocessableEntityError.code,
                "type": UnprocessableEntityError.error_type,
                "link": UnprocessableEntityError.link
            },
            response.body
        )
