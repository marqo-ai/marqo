import uuid
from starlette.middleware.base import BaseHTTPMiddleware

REQ_ID_HEADER = "x-request-id"

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        rid = request.headers.get(REQ_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = rid
        resp = await call_next(request)
        resp.headers[REQ_ID_HEADER] = rid
        return resp
