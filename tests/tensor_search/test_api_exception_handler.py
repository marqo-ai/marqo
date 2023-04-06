import pydantic
from unittest import mock
import pytest

from fastapi import Request
from starlette.datastructures import Headers
from starlette.types import Scope


from marqo.errors import InvalidArgError
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.api import validation_exception_handler

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


