from typing import Any, Dict, List, Optional

from pydantic import BaseModel, root_validator, Field


class Document(BaseModel):
    id: str
    fields: Dict[str, Any]


class GetDocumentResponse(BaseModel):
    path_id: str = Field(alias='pathId')
    document: Document

    @root_validator(pre=True)
    def build_document(cls, values):
        document_values = {
            'id': values.get('id'),
            'fields': values.get('fields')
        }
        del values['id']
        del values['fields']
        values['document'] = Document.parse_obj(document_values)
        return values


class GetBatchDocumentResponse(BaseModel):
    status: int
    path_id: str = Field(alias='pathId')
    id: Optional[str]  # used when status is not 200 and document is thus not returned
    document: Optional[Document]
    message: Optional[str]

    @root_validator(pre=True)
    def build_document(cls, values):
        if 'fields' in values:
            document_values = {
                'id': values.get('id'),
                'fields': values.get('fields')
            }
            del values['fields']
            values['document'] = Document.parse_obj(document_values)
        return values


class GetBatchResponse(BaseModel):
    responses: List[GetBatchDocumentResponse]
    errors: bool


class VisitDocumentsResponse(BaseModel):
    path_id: str = Field(alias='pathId')
    documents: List[Document]
    document_count: int = Field(alias='documentCount')
    continuation: Optional[str]
