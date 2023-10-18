"""
This module holds the classes which define the interface the delete documents
endpoint.
"""

import datetime
from typing import NamedTuple, Literal, List


class MqDeleteDocsResponse(NamedTuple):
    """An object that holds the data we send back to users"""
    index_name: str
    status_string: Literal["succeeded"]
    document_ids: List[str]
    deleted_docments_count: int
    success_list: List[dict]
    failure_list: List[dict]
    deletion_start: datetime.datetime
    deletion_end: datetime.datetime


class MqDeleteDocsRequest(NamedTuple):
    """An object that holds the data from users for a delete request"""
    index_name: str
    document_ids: List[str]
    auto_refresh: bool
