import datetime
from typing import List, NamedTuple, Literal
import json
from marqo import errors
from marqo._httprequests import HttpRequests
from marqo.config import Config
from marqo.tensor_search import validation, utils, enums


# -- Marqo delete endpoint interface: --


class MqDeleteDocsResponse(NamedTuple):
    index_name: str
    status_string: Literal["succeeded"]
    document_ids: List[str]
    deleted_docments_count: int
    deletion_start: datetime.datetime
    deletion_end: datetime.datetime


def format_delete_docs_response(marqo_response: MqDeleteDocsResponse) -> dict:
    return {
        "index_name": marqo_response.index_name, "status": marqo_response.status_string,
        "type": "documentDeletion", "details": {
            "receivedDocumentIds": len(marqo_response.document_ids),
            "deletedDocuments": marqo_response.deleted_docments_count,
        },
        "duration": utils.create_duration_string(marqo_response.deletion_end - marqo_response.deletion_start),
        "startedAt": utils.format_timestamp(marqo_response.deletion_start),
        "finishedAt": utils.format_timestamp(marqo_response.deletion_end),
    }


class MqDeleteDocsRequest(NamedTuple):
    index_name: str
    document_ids: List[str]
    auto_refresh: bool


def delete_documents(config: Config, del_request: MqDeleteDocsRequest) -> dict:
    """entrypoint function for deleting documents"""
    if not del_request.document_ids:
        raise errors.InvalidDocumentIdError("doc_ids can't be empty!")

    for _id in del_request.document_ids:
        validation.validate_id(_id)

    if config.backend == enums.SearchDb.opensearch:
        del_response: MqDeleteDocsResponse = delete_documents_marqo_os(config=config, deletion_instruction=del_request)
    else:
        raise RuntimeError(f"Config set to use unknown backend `{config.backend}`. "
                           f"See tensor_search.enums.SearchDB for allowed backends")

    return format_delete_docs_response(del_response)


# -- Marqo-OS-specific deletion implementation: --


def delete_documents_marqo_os(config: Config, deletion_instruction: MqDeleteDocsRequest) -> MqDeleteDocsResponse:
    """Deletes documents """

    # Prepare bulk delete request body
    bulk_request_body = ""
    for doc_id in deletion_instruction.document_ids:
        bulk_request_body += json.dumps({"delete": {"_index": deletion_instruction.index_name, "_id": doc_id}}) + "\n"

    # Send bulk delete request
    t0 = datetime.datetime.utcnow()
    delete_res_backend = HttpRequests(config=config).post(
        path="_bulk",
        body=bulk_request_body,
    )

    if deletion_instruction.auto_refresh:
        refresh_response = HttpRequests(config).post(path=f"{deletion_instruction.index_name}/_refresh")

    t1 = datetime.datetime.utcnow()
    deleted_documents_count = sum(1 for item in delete_res_backend["items"] if "delete" in item and item["delete"]["status"] == 200)

    mq_delete_res = MqDeleteDocsResponse(
        index_name=deletion_instruction.index_name, status_string='succeeded', document_ids=deletion_instruction.document_ids,
        deleted_docments_count=deleted_documents_count, deletion_start=t0,
        deletion_end=t1
    )
    return mq_delete_res
