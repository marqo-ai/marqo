from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field


# See https://docs.vespa.ai/en/reference/default-result-format.html
class RootFields(BaseModel):
    total_count: Optional[int] = Field(alias='totalCount')


class Degraded(BaseModel):
    adaptive_timeout: Optional[bool] = Field(alias='adaptive-timeout')
    match_phase: Optional[bool] = Field(alias='match-phase')
    non_ideal_state: Optional[bool] = Field(alias='non-ideal-state')
    timeout: Optional[bool]


class Coverage(BaseModel):
    coverage: int
    degraded: Optional[Degraded]
    documents: int
    full: bool
    nodes: int
    results: int
    results_full: int = Field(alias='resultsFull')


class Error(BaseModel):
    code: int
    summary: Optional[str]
    source: Optional[str]
    message: Optional[str]
    stack_trace: Optional[str] = Field(alias='stackTrace')
    transient: Optional[bool]


class AbstractChild(BaseModel):
    # label, value, and recursive children occur in aggregation results
    id: Optional[str]
    relevance: float
    source: Optional[str]
    label: Optional[str]
    value: Optional[str]
    coverage: Optional[Coverage]
    errors: Optional[List[Error]]
    children: Optional[List['Child']]


class Child(AbstractChild):
    fields: Optional[Dict[str, Any]]


class Root(AbstractChild):
    fields: Optional[RootFields]


class QueryResult(BaseModel):
    root: Root
    timing: Optional[Dict[str, Any]]
    trace: Optional[Dict[str, Any]]

    @property
    def hits(self) -> List[Child]:
        return self.root.children or []

    @property
    def total_count(self) -> int:
        return self.root.fields.total_count
