from marqo.tensor_search import constants
from marqo.tensor_search import validation
from marqo._httprequests import HttpRequests
from marqo.config import Config
from marqo import errors
from marqo.tensor_search.tensor_search_logging import get_logger

logger = get_logger(__name__)


def convert_watermark_to_bytes(watermark: str, total_in_bytes: str = None):
    """
    Converts a value to bytes.
    It could possible be:
    1. Bytes (eg 123.4gb) - do nothing
    2. MB, GB, TB, etc, (eg 123.4gb) - divide by some power of 1024 to get bytes
    3. Ratio (e.g. 0.9) - multiply by total_in_bytes to get bytes
    4. Percentage (e.g. 90%) - convert to ratio then multiply by total_in_bytes to get bytes

    Returns: watermark in bytes (float)
    """

    # Initial validation
    if watermark is None:
        raise errors.InternalError("OpenSearch disk watermark cannot be None.")
    watermark = watermark.replace(" ", "")
    if watermark == "":
        raise errors.InternalError("OpenSearch disk watermark cannot be empty string.")
    
    if total_in_bytes is not None:
        total_in_bytes = validation.validate_nonnegative_number(total_in_bytes, "OpenSearch disk total size in bytes")
    
    if watermark[-2:].lower() in constants.BYTE_SUFFIXES:
        # Watermark in KB/MB/GB/TB format
        # Bytes represent MIN disk space AVAILABLE
        # TODO: Do we use 1000 or 1024 for conversion?
        numeric_watermark = validation.validate_nonnegative_number(watermark[:-2], "OpenSearch disk watermark value")
        multiplier = 1024 ** constants.BYTE_SUFFIXES.index(watermark[-2:].lower())
        return numeric_watermark * multiplier
        
    elif watermark[-1].lower() == "b":
        # Watermark in BYTE format
        # Bytes represent MIN disk space AVAILABLE
        numeric_watermark = validation.validate_nonnegative_number(watermark[:-1], "OpenSearch disk watermark value")
        return numeric_watermark
    
    # Percentage or Ratio calculation
    if watermark[-1] == "%":
        # Watermark in PERCENTAGE format
        # Ratio & percentage represent MAX disk space USED
        numeric_watermark = validation.validate_nonnegative_number(watermark[:-1], "OpenSearch disk watermark value")
        ratio_watermark = (100 - numeric_watermark) / 100
    else:
        # Watermark in RATIO format
        # Ratio & percentage represent MAX disk space USED
        numeric_watermark = validation.validate_nonnegative_number(watermark, "OpenSearch disk watermark value")
        ratio_watermark = 1 - numeric_watermark

    if ratio_watermark < 0 or ratio_watermark > 1:
        raise errors.InternalError("OpenSearch watermark ratio or percentage cannot be negative or more than 100%.")
    if total_in_bytes is None:
        raise errors.InternalError("total_in_bytes must be provided for ratio or percentage watermark.")
    
    return total_in_bytes * ratio_watermark


def check_opensearch_disk_watermark_breach(config: Config):
    """
    Checks if the OpenSearch disk watermark is breached:
    1. Check disk watermark from the settings endpoint.
      - Check transient, persistent, then default settings
      - convert it to a size in BYTES (it could initially be a percentage, ratio, or size in B, GB, MB, etc.)
    2. Check the current available disk space from the stats endpoint.
    3. Compare current avilable space to watermark value.

    Returns: red if watermark is breached, green otherwise.
    """

    # TODO: Check for disk threshhold enabled?
    # Query opensearch for watermark
    raw_flood_stage_watermark = None
    for settings_type in constants.OPENSEARCH_CLUSTER_SETTINGS_TYPES:
        try:
            # Check for transient, persistent, then defaults settings
            opensearch_disk_settings = HttpRequests(config).get(path="_cluster/settings?include_defaults=true")\
                [settings_type]["cluster"]["routing"]["allocation"]["disk"]
            # TODO: Should we use flood stage? or high? or low?
            raw_flood_stage_watermark = opensearch_disk_settings["watermark"]["flood_stage"]
            logger.debug(f"Found disk flood stage watermark in {settings_type} settings: {raw_flood_stage_watermark}")
            break
        except KeyError:
            logger.debug(f"Flood stage watermark not found in {settings_type} settings.")
    
    if not raw_flood_stage_watermark:
        raise errors.BackendCommunicationError("Could not find disk flood stage watermark in OpenSearch settings.")
    
    # Query opensearch for disk space
    filesystem_stats = HttpRequests(config).get(path="_cluster/stats")["nodes"]["fs"]
    minimum_available_disk_space = convert_watermark_to_bytes(watermark=raw_flood_stage_watermark, total_in_bytes=filesystem_stats["total_in_bytes"])

    if filesystem_stats["available_in_bytes"] < minimum_available_disk_space:
        return "red"
    return "green"