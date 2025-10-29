import uvicorn
from fastapi import FastAPI

from .api.exception_handlers import register_exception_handlers
from .api.lifespan import lifespan
from .api.request_id import RequestIdMiddleware
from .api.v1_routes import router
from .version import get_version

app = FastAPI(
    title="Marqo Model Management Container", lifespan=lifespan, version=get_version()
)
app.add_middleware(RequestIdMiddleware)
register_exception_handlers(app)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8883)
