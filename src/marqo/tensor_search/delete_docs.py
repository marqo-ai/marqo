import datetime
from typing import List

from marqo import errors
from marqo._httprequests import HttpRequests
from marqo.config import Config
from marqo.tensor_search import validation, utils


def delete_documents(config: Config, index_name: str, doc_ids: List[str], auto_refresh):
    """Deletes documents """
    if not doc_ids:
        raise errors.InvalidDocumentIdError("doc_ids can't be empty!")

    for _id in doc_ids:
        validation.validate_id(_id)

    # TODO: change to timer()
    t0 = datetime.datetime.utcnow()
    delete_res_backend = HttpRequests(config=config).post(
        path=f"{index_name}/_delete_by_query", body={
            "query": {
                "terms": {
                    "_id": doc_ids
                }
            }
        }
    )
    if auto_refresh:
        refresh_response = HttpRequests(config).post(path=F"{index_name}/_refresh")
    t1 = datetime.datetime.utcnow()
    delete_res = {
        "index_name": index_name, "status": "succeeded",
        "type": "documentDeletion", "details": {
            "receivedDocumentIds": len(doc_ids),
            "deletedDocuments": delete_res_backend["deleted"],
        },
        "duration": utils.create_duration_string(t1 - t0),
        "startedAt": utils.format_timestamp(t0),
        "finishedAt": utils.format_timestamp(t1),
    }
    return delete_res
