from fastapi import Request
from fastapi.routing import APIRoute

from marqo.logging import get_logger

logger = get_logger(__name__)


class MarqoCustomRoute(APIRoute):
    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def marqo_custom_route_handler(request: Request):
            try:
                return await original_route_handler(request)
            except Exception as exc:
                await self._log_the_error(request, exc)
                raise exc

        return marqo_custom_route_handler

    async def _log_the_error(self, request: Request, exc: Exception):
        if self._is_log_stack_trace(exc):
            logger.error(str(exc), exc_info=True)
        if self._is_log_request_body(exc):
            body_bytes = await request.body()
            logger.error(f"Request body: {body_bytes.decode('utf-8')}")

    def _is_log_stack_trace(self, exc) -> bool:
        return True

    def _is_log_request_body(self, exc) -> bool:
        return True

