from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field


class RootFields(BaseModel):
    total_count: int = Field(alias='totalCount')


class Degraded(BaseModel):
    adaptive_timeout: bool = Field(alias='adaptive-timeout')
    match_phase: bool = Field(alias='match-phase')
    non_ideal_state: bool = Field(alias='non-ideal-state')
    timeout: bool


class Coverage(BaseModel):
    coverage: int
    degraded: Optional[Degraded]
    documents: int
    full: bool
    nodes: int
    results: int
    results_full: int = Field(alias='resultsFull')


class Child(BaseModel):
    # label, value, and recursive children occur in aggregation results
    id: str
    relevance: float
    source: Optional[str]
    label: Optional[str]
    value: Optional[str]
    fields: Optional[Dict[str, Any]]
    children: Optional[List['Child']]


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
