"""Classes used for API communication"""
from pydantic import BaseModel


class SearchQuery(BaseModel):
    q: str


class AddDocuments(BaseModel):
    docs: list
    index_name: str
