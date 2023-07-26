from typing import Optional, Tuple
import copy

from marqo.config import Config
from marqo import errors
from marqo._httprequests import HttpRequests
from marqo.tensor_search.enums import HealthStatuses


def generate_heath_check_response(config: Config, index_name: Optional[str] = None) -> dict:
    """Generate the health check response for check_heath(), check_index_health() APIs in tensor_search"""
    marqo_status = get_marqo_status()
    marqo_os_status = get_marqo_os_status(config, index_name=index_name)
    marqo_status, marqo_os_status = aggregate_status(marqo_status, marqo_os_status)

    return {
        "status": marqo_status,
        "backend": {
            "status": marqo_os_status
        }
    }


def get_marqo_status() -> str:
    """Check the Marqo instance status."""
    return HealthStatuses.green


def get_marqo_os_status(config: Config, index_name: Optional[str] = None) -> str:
    """Check the Marqo-os backend status."""
    TIMEOUT = 3
    marqo_os_health_check_response = None
    path = f"_cluster/health/{index_name}" if index_name else "_cluster/health"

    try:
        timeout_config = copy.deepcopy(config)
        timeout_config.timeout = TIMEOUT
        marqo_os_health_check_response = HttpRequests(timeout_config).get(path=path)
    except errors.BackendCommunicationError:
        marqo_os_health_check_response = None

    if marqo_os_health_check_response is not None:
        if "status" in marqo_os_health_check_response and marqo_os_health_check_response['status'] \
                in list(HealthStatuses):
            marqo_os_status = HealthStatuses[marqo_os_health_check_response['status']]
        else:
            marqo_os_status = HealthStatuses.red
    else:
        marqo_os_status = HealthStatuses.red

    return marqo_os_status


def aggregate_status(marqo_status: str, marqo_os_status: str) -> Tuple[str, str]:
    """Aggregate the Marqo instance and Marqo-os backend status."""
    marqo_status = max(marqo_status, marqo_os_status)
    return marqo_status, marqo_os_status
