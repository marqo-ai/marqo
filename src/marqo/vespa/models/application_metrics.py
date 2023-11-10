from typing import List, Any, Dict

from pydantic import BaseModel


class Status(BaseModel):
    code: str
    description: str


class Service(BaseModel):
    name: str
    timestamp: int
    status: Status
    metrics: List[Dict[str, Any]]


class Node(BaseModel):
    hostname: str
    role: str
    services: List[Service]


class ApplicationMetrics(BaseModel):
    nodes: List[Node]
