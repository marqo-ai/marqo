import uuid

from starlette.middleware.base import BaseHTTPMiddleware

REQ_ID_HEADER = "x-request-id"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    A middleware that assigns a unique request ID to each incoming HTTP request.
    The request ID is added to the request state and also included in the response headers.

    This is useful for tracing and debugging requests.
    """

    async def dispatch(self, request, call_next):
        rid = request.headers.get(REQ_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = rid
        resp = await call_next(request)
        resp.headers[REQ_ID_HEADER] = rid
        return resp
