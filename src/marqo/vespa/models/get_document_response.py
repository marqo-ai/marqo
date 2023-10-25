from typing import Any, Dict, List, Optional

from pydantic import BaseModel, root_validator


class Document(BaseModel):
    id: str
    fields: Dict[str, Any]


class GetDocumentResponse(BaseModel):
    pathId: str
    document: Document

    @root_validator(pre=True)
    def build_document(cls, values):
        document_values = {
            "id": values.get("id"),
            "fields": values.get("fields")
        }
        values["document"] = Document.parse_obj(document_values)
        return values


class BatchGetDocumentResponse(BaseModel):
    pathId: str
    documents: List[Document]
    documentCount: int
    continuation: Optional[str]
