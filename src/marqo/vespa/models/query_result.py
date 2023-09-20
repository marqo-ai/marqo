from typing import List

from pydantic import BaseModel


class RootFields(BaseModel):
    totalCount: int


class ChildFields(BaseModel):
    sddocname: str = None
    documentid: str = None

    class Config:
        # Let pydantic add all document fields
        extra = "allow"


class Coverage(BaseModel):
    coverage: int
    documents: int
    full: bool
    nodes: int
    results: int
    resultsFull: int


class Child(BaseModel):
    id: str
    relevance: float
    source: str = None
    fields: ChildFields


class Root(BaseModel):
    id: str
    relevance: float
    fields: RootFields
    coverage: Coverage
    children: List[Child]


class QueryResult(BaseModel):
    root: Root
