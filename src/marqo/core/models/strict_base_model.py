from pydantic import BaseModel


class StrictBaseModel(BaseModel):
    class Config:
        extra: str = "forbid"
