"""
This module handles the delete documents endpoint
"""
import datetime

from marqo.config import Config
from marqo.tensor_search import validation, utils, enums
from marqo.tensor_search.models.delete_docs_objects import MqDeleteDocsResponse, MqDeleteDocsRequest
from marqo.tensor_search.tensor_search_logging import get_logger

logger = get_logger(__name__)


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

    if config.backend == enums.SearchDb.vespa:
        del_response: MqDeleteDocsResponse = delete_documents_vespa(config=config, deletion_instruction=del_request)
    else:
        raise RuntimeError(f"Config set to use unknown backend `{config.backend}`. "
                           f"See tensor_search.enums.SearchDB for allowed backends")

    return format_delete_docs_response(del_response)


# -- Marqo-OS-specific deletion implementation: --


def delete_documents_vespa(config: Config, deletion_instruction: MqDeleteDocsRequest) -> MqDeleteDocsResponse:
    """Deletes documents """
    t0 = datetime.datetime.utcnow()
    responses = config.vespa_client.delete_batch(deletion_instruction.document_ids, deletion_instruction.schema_name)
    t1 = datetime.datetime.utcnow()

    deleted_documents_count = 0
    result_list = []
    for response in responses.responses:
        if response.status == 200:
            deleted_documents_count += 1
            result_list.append(
                {
                    '_id': _get_id_from_vespa_id(response.id),
                    'status': 200,
                    'result': 'deleted'
                }
            )
        elif response.status == 404:
            # Note: Vespa returns 200 even when document does not exist. So this case may never be reached
            result_list.append(
                {
                    '_id': _get_id_from_vespa_id(response.id),
                    'status': 404,
                    'result': 'not_found'
                }
            )
        else:
            result_list.append(
                {
                    '_id': _get_id_from_vespa_id(response.id),
                    'status': response.status,
                    'result': 'error'
                }
            )
            logger.error(f'Failed to delete document: {response}')

    mq_delete_res = MqDeleteDocsResponse(
        index_name=deletion_instruction.index_name, status_string='succeeded',
        document_ids=deletion_instruction.document_ids,
        deleted_documents_count=deleted_documents_count, deletion_start=t0,
        deletion_end=t1, result_list=result_list
    )
    return mq_delete_res


def _get_id_from_vespa_id(vespa_id: str) -> str:
    """Returns the document ID from a Vespa ID. Vespa IDs are of the form `namespace::document_id`."""
    return vespa_id.split('::')[-1]
