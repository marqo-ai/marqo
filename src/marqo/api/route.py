from fastapi import Request
from fastapi.routing import APIRoute

from marqo.logging import get_logger

logger = get_logger(__name__)


class MarqoCustomRoute(APIRoute):
    """This is a custom route that logs the error and raises it.

    The log will include the stack trace of the error for debugging purposes.
    The raised error will be handled by the exception handlers. We DO NOT handle the error here.
    """
    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def marqo_custom_route_handler(request: Request):
            try:
                return await original_route_handler(request)
            except Exception as exc:
                logger.error(str(exc), exc_info=True)
                raise exc

        return marqo_custom_route_handler