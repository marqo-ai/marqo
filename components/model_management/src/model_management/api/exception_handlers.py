import json
from typing import Any, Dict, List

import model_management.errors.http_errors as http_errors
import model_management.services.errors as service_errors
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.responses import Response

from ..contracts.problem import Problem
from ..errors.base import AppError


def _map_service_errors_to_http_errors(
    service_exception: service_errors.ServiceError,
) -> AppError:
    """
    Map service-layer errors to appropriate HTTP errors.
    """
    service_to_http_error_map = [
        ((service_errors.ModelDownloadFailedError,), http_errors.InvalidArgumentError),
        (
            (service_errors.ModelOperationInProgressError,),
            http_errors.OperationConflictError,
        ),
        (
            (service_errors.TritonCommunicationError,),
            http_errors.DependencyBadGatewayError,
        ),
        ((service_errors.InternalServerError,), http_errors.InternalServerError),
    ]

    for service_error_classes, http_error_class in service_to_http_error_map:
        if isinstance(service_exception, service_error_classes):
            return http_error_class(service_exception.message)

    return http_errors.InternalServerError(service_exception.message)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(service_errors.ServiceError, _service_error_handler)
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(Exception, _catch_all_handler)


def _problem_response(request: Request, exc: AppError) -> Response:
    """
    Convert an AppError into a RFC 7807 Problem+JSON response.
    This assumes the AppError has http_status, code, and detail attributes.
    You can extend the Problem model with additional fields by adding an 'extras' dict to your
    AppError subclass.

    :param request: The incoming request
    :param exc: The raised AppError
    :return:
        A JSONResponse with Problem+JSON content type and appropriate status code in
        RFC 7807 format.
    """
    body = Problem(
        title=str(exc.__class__.__name__),
        status=exc.http_status,
        code=exc.code,
        detail=str(exc),
        instance=str(request.url),
        request_id=getattr(request.state, "request_id", None),
        extras=getattr(exc, "extras", None),
    ).model_dump()
    return JSONResponse(
        body, status_code=exc.http_status, media_type="application/problem+json"
    )


async def _validation_error_handler(
    request: Request, exc: RequestValidationError
) -> Response:
    """
    Normalise the FastAPI/Pydantic validation errors into a tidy, client-friendly payload.

    :param request: the incoming request
    :param exc: the raised RequestValidationError or ValidationError
    :return: A JSONResponse with Problem+JSON content type and 400 status code.
    """
    error_messages = [
        {
            "loc": error.get("loc", ""),
            "msg": error.get("msg", ""),
            "type": error.get("type", ""),
        }
        for error in exc.errors()
    ]

    return _problem_response(
        request,
        http_errors.InvalidArgumentError(
            message=json.dumps(error_messages),
        ),
    )


async def _service_error_handler(
    request: Request, exc: service_errors.ServiceError
) -> Response:
    """
    Catch transport-agnostic service/domain errors and map them to HTTP-aware AppErrors.
    """
    app_err = _map_service_errors_to_http_errors(exc)
    # (Optional) log original exception & stacktrace here
    return _problem_response(request, app_err)


async def _app_error_handler(request: Request, exc: AppError) -> Response:
    return _problem_response(request, exc)


async def _catch_all_handler(request: Request, exc: Exception) -> Response:
    return _problem_response(
        request, http_errors.InternalServerError("An unexpected error occurred.")
    )


def _normalize_validation_errors(exc: RequestValidationError) -> Dict[str, Any]:
    """
    Shape FastAPI/Pydantic validation errors into a tidy, client-friendly payload.
    """
    raw = exc.errors()
    errors: List[Dict[str, Any]] = []

    for e in raw:
        loc = e.get("loc", [])
        # strip top-level sources like 'body'|'query'|'path' for a cleaner field path
        stripped = [
            str(p)
            for p in loc
            if p not in ("body", "query", "path", "header", "cookie")
        ]
        field_path = ".".join(stripped)

        message = e.get("msg") or e.get("message") or "Invalid value"
        type_ = e.get("type")  # e.g. value_error.missing
        # pydantic v2 may include these:
        input_ = e.get("input")
        ctx = e.get("ctx")

        entry = {
            "location": loc,  # full original location
            "field": field_path or None,
            "message": message,
            "type": type_,
            "input": input_,
            "ctx": ctx,
        }
        # drop Nones for a cleaner JSON
        errors.append({k: v for k, v in entry.items() if v is not None})

    # convenience: map of field -> list of messages (nice for forms/clients)
    field_errors: Dict[str, List[str]] = {}
    for e in errors:
        fld = e.get("field")
        if fld:
            field_errors.setdefault(fld, []).append(e["message"])

    return {
        "errors": errors,  # detailed, per-violation list
        "field_errors": field_errors,  # quick lookup by field
        "count": len(errors),
    }
