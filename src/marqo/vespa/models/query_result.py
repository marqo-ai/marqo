from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field


class RootFields(BaseModel):
    total_count: int = Field(alias='totalCount')


class Coverage(BaseModel):
    coverage: int
    documents: int
    full: bool
    nodes: int
    results: int
    results_full: int = Field(alias='resultsFull')


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
    trace: Optional[Dict[str, Any]]

    @property
    def hits(self) -> List[Child]:
        return self.root.children

    @property
    def total_count(self) -> int:
        return self.root.fields.total_count
