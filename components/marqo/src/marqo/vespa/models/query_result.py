from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field


class MarqoFields(BaseModel):
    """
    Fields that collect the metadata from the Custom Searcher.
    """
    sort_candidates: Optional[int] = Field(None, alias='sortCandidates')
    relevant_candidates: Optional[int] = Field(None, alias='relevantCandidates')
    probe_candidates: Optional[int] = Field(None, alias='probeCandidates')
    post_process_candidates: Optional[int] = Field(None, alias='postProcessCandidates')


# See https://docs.vespa.ai/en/reference/default-result-format.html
class RootFields(BaseModel):
    total_count: Optional[int] = Field(None, alias='totalCount')
    marqo_fields: Optional[MarqoFields] = Field(None, alias='marqo__fields')


class Degraded(BaseModel):
    adaptive_timeout: Optional[bool] = Field(None, alias='adaptive-timeout')
    match_phase: Optional[bool] = Field(None, alias='match-phase')
    non_ideal_state: Optional[bool] = Field(None, alias='non-ideal-state')
    timeout: Optional[bool] = None


class Coverage(BaseModel):
    coverage: int
    degraded: Optional[Degraded] = None
    documents: int
    full: bool
    nodes: int
    results: int
    results_full: int = Field(alias='resultsFull')


class Error(BaseModel):
    code: int
    summary: Optional[str] = None
    source: Optional[str] = None
    message: Optional[str] = None
    stack_trace: Optional[str] = Field(None, alias='stackTrace')
    transient: Optional[bool] = None


class AbstractChild(BaseModel):
    # label, value, and recursive children occur in aggregation results
    id: Optional[str] = None
    relevance: float
    source: Optional[str] = None
    label: Optional[str] = None
    value: Optional[str] = None
    coverage: Optional[Coverage] = None
    errors: Optional[List[Error]] = None
    children: Optional[List['Child']] = None


class Child(AbstractChild):
    fields: Optional[Dict[str, Any]] = None


class Root(AbstractChild):
    fields: Optional[RootFields] = None


class QueryResult(BaseModel):
    root: Root
    timing: Optional[Dict[str, Any]] = None
    trace: Optional[Dict[str, Any]] = None

    @property
    def hits(self) -> List[Child]:
        return self.root.children or []

    @property
    def total_count(self) -> int:
        return self.root.fields.total_count

    @property
    def facets(self) -> List[Child]:
        return self.root.children or []
