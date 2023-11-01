"""
This module handles the delete documents endpoint
"""
import datetime
import json
from marqo._httprequests import HttpRequests
from marqo.config import Config
from marqo.tensor_search import validation, utils, enums
from marqo.tensor_search.models.delete_docs_objects import MqDeleteDocsResponse, MqDeleteDocsRequest
import pprint
from marqo import errors

# -- Marqo delete endpoint interface: --


def format_delete_docs_response(marqo_response: MqDeleteDocsResponse) -> dict:
    """This formats the delete response for users """
    return {
        "index_name": marqo_response.index_name, "status": marqo_response.status_string,
        "type": "documentDeletion", 
        "items": marqo_response.result_list,
        "details": {
            "receivedDocumentIds": len(marqo_response.document_ids),
            "deletedDocuments": marqo_response.deleted_documents_count,
        },
        "duration": utils.create_duration_string(marqo_response.deletion_end - marqo_response.deletion_start),
        "startedAt": utils.format_timestamp(marqo_response.deletion_start),
        "finishedAt": utils.format_timestamp(marqo_response.deletion_end),
    }


# -- Data-layer agnostic logic --


def delete_documents(config: Config, del_request: MqDeleteDocsRequest) -> dict:
    """entrypoint function for deleting documents"""

    validation.validate_delete_docs_request(
        delete_request=del_request,
        max_delete_docs_count=utils.read_env_vars_and_defaults_ints(enums.EnvVars.MARQO_MAX_DELETE_DOCS_COUNT)
    )

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

    deleted_documents_count = 0
    result_list = []

    for item in delete_res_backend["items"]:
        if "delete" in item:
            try:
                result_list.append({
                    "_id": item["delete"]["_id"],
                    "_shards": item["delete"]["_shards"],
                    "status": item["delete"]["status"],
                    "result": item["delete"]["result"],
                })
                if item["delete"]["status"] == 200:
                    deleted_documents_count += 1
            except KeyError as e:
                raise errors.InternalError(f"Failed to parse delete response from backend. "
                                           f"Missing key in response. Reason: {e}")


    mq_delete_res = MqDeleteDocsResponse(
        index_name=deletion_instruction.index_name, status_string='succeeded', document_ids=deletion_instruction.document_ids,
        deleted_documents_count=deleted_documents_count, deletion_start=t0,
        deletion_end=t1, result_list=result_list
    )
    return mq_delete_res
