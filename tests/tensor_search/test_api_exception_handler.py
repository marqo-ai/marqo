import pydantic
from unittest import mock
import pytest

from fastapi import Request
from starlette.datastructures import Headers
from starlette.types import Scope

from marqo import exceptions as base_exceptions
from marqo.api import exceptions as api_exceptions
from marqo.core import exceptions as core_exceptions
from marqo.vespa import exceptions as vespa_exceptions
from marqo.api.exceptions import InvalidArgError

from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.api import validation_exception_handler, marqo_base_exception_handler

class BaseMarqoModel(pydantic.BaseModel):
    class Config:
        extra: str = "forbid"
    pass

class TestApiValidation(MarqoTestCase):
    
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


    async def test_validation_exception_handler_valid_error(self):
        error = pydantic.ValidationError(errors=[{'loc': ('field',), 'msg': 'error message', 'type': 'value_error'}], model=BaseMarqoModel)
        response = validation_exception_handler(self.normal_request, error)
        assert response.status_code == InvalidArgError.status_code
        assert response.json() == {
            "message": '[{"loc": ["field"], "msg": "error message", "type": "value_error"}]',
            "code": InvalidArgError.code,
            "type": InvalidArgError.error_type,
            "link": InvalidArgError.link
        }

    async def test_validation_exception_handler_empty_error(self):
        error = pydantic.ValidationError(errors=[], model=BaseMarqoModel)
        response = validation_exception_handler(self.normal_request, error)
        assert response.status_code == InvalidArgError.status_code
        assert response.json() == {
            "message": '[]',
            "code": InvalidArgError.code,
            "type": InvalidArgError.error_type,
            "link": InvalidArgError.link
        }

    async def test_validation_exception_handler_multiple_errors(self):
        error = pydantic.ValidationError(errors=[
            {'loc': ('field1',), 'msg': 'error message 1', 'type': 'value_error'},
            {'loc': ('field2',), 'msg': 'error message 2', 'type': 'value_error'},
            {'loc': ('field3',), 'msg': 'error message 3', 'type': 'value_error'}
        ], model=BaseMarqoModel)
        response = validation_exception_handler(self.normal_request, error)
        assert response.status_code == InvalidArgError.status_code
        assert response.json() == {
            "message": '[{"loc": ["field1"], "msg": "error message 1", "type": "value_error"}, '
                    '{"loc": ["field2"], "msg": "error message 2", "type": "value_error"}, '
                    '{"loc": ["field3"], "msg": "error message 3", "type": "value_error"}]',
            "code": InvalidArgError.code,
            "type": InvalidArgError.error_type,
            "link": InvalidArgError.link
        }

    async def test_validation_exception_handler_missing_fields_loc(self):
        error = pydantic.ValidationError(errors=[{'msg': 'error message', 'type': 'value_error'}], model=BaseMarqoModel)
        response = validation_exception_handler(self.normal_request, error)
        assert response.status_code == InvalidArgError.status_code
        assert response.json() == {
            "message": '[{"loc": "", "msg": "error message", "type": "value_error"}]',
            "code": InvalidArgError.code,
            "type": InvalidArgError.error_type,
            "link": InvalidArgError.link
        }

    async def test_validation_exception_handler_missing_fields_msg(self):
        error = pydantic.ValidationError(errors=[{'loc': ('field',), 'type': 'value_error'}], model=BaseMarqoModel)
        response = validation_exception_handler(self.normal_request, error)
        assert response.status_code == InvalidArgError.status_code
        assert response.json() == {
            "message": '[{"loc": ["field"], "msg": "", "type": "value_error"}]',
            "code": InvalidArgError.code,
            "type": InvalidArgError.error_type,
            "link": InvalidArgError.link
        }

    async def test_validation_exception_handler_missing_fields_type(self):
        error = pydantic.ValidationError(errors=[{'loc': ('field',), 'msg': 'error message'}], model=BaseMarqoModel)
        response = validation_exception_handler(self.normal_request, error)
        assert response.status_code == InvalidArgError.status_code
        assert response.json() == {
            "message": '[{"loc": ["field"], "msg": "error message", "type": ""}]',
            "code": InvalidArgError.code,
            "type": InvalidArgError.error_type,
            "link": InvalidArgError.link
        }

    async def test_validation_exception_handler_different_error_types(self):
        errors = [
            pydantic.ValidationError(errors=[{'loc': ('field',), 'msg': 'type error', 'type': 'type_error.integer'}], model=BaseMarqoModel),
            pydantic.ValidationError(errors=[{'loc': ('field',), 'msg': 'value error', 'type': 'value_error'}], model=BaseMarqoModel),
            pydantic.ValidationError(errors=[{'loc': ('field',), 'msg': 'custom error', 'type': 'custom_error'}], model=BaseMarqoModel)
        ]
        for error in errors:
            response = validation_exception_handler(self.normal_request, error)
            assert response.status_code == InvalidArgError.status_code
            assert response.json() == {
                "message": f'[{{"loc": ["field"], "msg": "{error.errors[0]["msg"]}", "type": "{error.errors[0]["type"]}"}}]',
                "code": InvalidArgError.code,
                "type": InvalidArgError.error_type,
                "link": InvalidArgError.link
            }

    async def test_validation_exception_handler_non_validation_error(self):
        error = ValueError('some error')
        with pytest.raises(ValueError):
            validation_exception_handler(self.normal_request, error)

    async def test_validation_exception_handler_custom_error_message(self):
        error = pydantic.ValidationError(errors=[{'loc': ('field',), 'msg': 'custom error message', 'type': 'value_error'}], model=BaseMarqoModel)
        response = validation_exception_handler(self.normal_request, error)
        assert response.status_code == InvalidArgError.status_code
        assert response.json() == {
            "message": '[{"loc": ["field"], "msg": "custom error message", "type": "value_error"}]',
            "code": InvalidArgError.code,
            "type": InvalidArgError.error_type,
            "link": InvalidArgError.link
        }

    async def test_validation_exception_handler_different_request_objects(self):

        # valid request object
        request = mock.MagicMock()
        request.method = 'POST'
        request.headers = {'content-type': 'application/json'}
        request.url = 'https://example.com'
        request.json = lambda: {"field": "value"}

        error = pydantic.ValidationError(errors=[{'loc': ('field',), 'msg': 'error message', 'type': 'value_error'}], model=BaseMarqoModel)
        response = validation_exception_handler(request, error)
        assert response.status_code == InvalidArgError.status_code

        # invalid request object with missing json method
        request = mock.MagicMock()
        request.method = 'POST'
        request.headers = {'content-type': 'application/json'}
        request.url = URL('https://example.com')
        with pytest.raises(AttributeError):
            validation_exception_handler(request, error)

        # invalid request object with missing headers
        request = mock.MagicMock()
        request.method = 'POST'
        request.url = URL('https://example.com')
        with pytest.raises(AttributeError):
            validation_exception_handler(request, error)

    async def test_validation_exception_handler_response_content(self):
        error = pydantic.ValidationError(errors=[{'loc': ('field',), 'msg': 'error message', 'type': 'value_error'}], model=BaseMarqoModel)
        response = validation_exception_handler(self.normal_request, error)
        content = response.json()
        assert 'message' in content
        assert 'code' in content
        assert 'type' in content
        assert 'link' in content

    async def test_validation_exception_handler_response_status_code(self):
        error = pydantic.ValidationError(errors=[{'loc': ('field',), 'msg': 'error message', 'type': 'value_error'}], model=BaseMarqoModel)
        response = validation_exception_handler(self.normal_request, error)
        assert response.status_code == InvalidArgError.status_code


class TestBaseExceptionHandler(MarqoTestCase):
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

    def test_base_exception_handler_base_errors(self):
        with self.subTest("Base error: Internal Error"):
            try:
                marqo_base_exception_handler(self.normal_request, base_exceptions.InternalError(self.generic_error_message))
                raise AssertionError
            except api_exceptions.InternalError as e:
                assert e.message.startswith(self.generic_error_message)     # 500 error

        with self.subTest("Base error: Invalid Argument Error"):
            try:
                marqo_base_exception_handler(self.normal_request, base_exceptions.InvalidArgumentError(self.generic_error_message))
                raise AssertionError
            except api_exceptions.InvalidArgError as e:
                assert e.message == self.generic_error_message   # 400 error


    def test_base_exception_handler_core_errors(self):
        with self.subTest("Core error: Index Exists Error"):
            try:
                marqo_base_exception_handler(self.normal_request, core_exceptions.IndexExistsError(self.generic_error_message))
                raise AssertionError
            except api_exceptions.IndexAlreadyExistsError as e:
                assert e.message == self.generic_error_message

        with self.subTest("Core error: Index Not Found Error"):
            try:
                marqo_base_exception_handler(self.normal_request, core_exceptions.IndexNotFoundError(self.generic_error_message))
                raise AssertionError
            except api_exceptions.IndexNotFoundError as e:
                assert e.message == self.generic_error_message

        with self.subTest("Core error: Parsing Error"):
            try:
                marqo_base_exception_handler(self.normal_request, core_exceptions.ParsingError(self.generic_error_message))
                raise AssertionError
            except api_exceptions.MarqoWebError as e:
                assert e.message.startswith(self.generic_error_message)     # Generic 500

        with self.subTest("Core error: Vespa Document Parsing Error"):
            try:
                marqo_base_exception_handler(self.normal_request, core_exceptions.VespaDocumentParsingError(self.generic_error_message))
                raise AssertionError
            except api_exceptions.BackendDataParsingError as e:
                assert e.message.startswith(self.generic_error_message)      # 500

        with self.subTest("Core error: Marqo Document Parsing Error"):
            try:
                marqo_base_exception_handler(self.normal_request, core_exceptions.MarqoDocumentParsingError(self.generic_error_message))
                raise AssertionError
            except api_exceptions.InvalidArgError as e:
                assert e.message == self.generic_error_message

        with self.subTest("Core error: Invalid Data Type Error"):
            try:
                marqo_base_exception_handler(self.normal_request, core_exceptions.InvalidDataTypeError(self.generic_error_message))
                raise AssertionError
            except api_exceptions.InvalidArgError as e:
                assert e.message == self.generic_error_message

        with self.subTest("Core error: Invalid Field Name Error"):
            try:
                marqo_base_exception_handler(self.normal_request, core_exceptions.InvalidFieldNameError(self.generic_error_message))
                raise AssertionError
            except api_exceptions.InvalidFieldNameError as e:
                assert e.message == self.generic_error_message

        with self.subTest("Core error: Filter String Parsing Error"):
            try:
                marqo_base_exception_handler(self.normal_request, core_exceptions.FilterStringParsingError(self.generic_error_message))
                raise AssertionError
            except api_exceptions.InvalidArgError as e:
                assert e.message == self.generic_error_message

    def test_base_exception_handler_vespa_errors(self):
        with self.subTest("Vespa error: Vespa Error"):
            try:
                marqo_base_exception_handler(self.normal_request, vespa_exceptions.VespaError(self.generic_error_message))
                raise AssertionError
            except api_exceptions.MarqoWebError as e:
                assert e.message.startswith(self.generic_error_message)    # Generic 500