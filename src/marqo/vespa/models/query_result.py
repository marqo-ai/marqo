from typing import List, Dict, Any

from pydantic import BaseModel


class RootFields(BaseModel):
    totalCount: int


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
    fields: Dict[str, Any]


class Root(BaseModel):
    id: str
    relevance: float
    fields: RootFields
    coverage: Coverage
    children: List[Child] = []


class QueryResult(BaseModel):
    root: Root
