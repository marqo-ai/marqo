from typing import List, Optional, Any, Dict, Set

from pydantic import Field, root_validator

from marqo.base_model import MarqoBaseModel


class BatchResponseStats(MarqoBaseModel):
    success_count: int = Field(default=0)
    error_count: int = Field(default=0)
    failure_count: int = Field(default=0)

    def get_header_dict(self) -> Dict[str, str]:
        return {
            "x-count-success": str(self.success_count),
            "x-count-failure": str(self.failure_count),
            "x-count-error": str(self.error_count),
        }


class MarqoBaseDocumentsResponse(MarqoBaseModel):
    """A base documents API response model."""
    _batch_response_stats: BatchResponseStats = Field(exclude=True, default_factory=BatchResponseStats)

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Setting default exclude to exclude _batch_response_stats from the response.

        Setting Field(exclude=True) does not work for
        _batch_response_stats: BatchResponseStats = Field(exclude=True, default_factory=BatchResponseStats). So we need
        to exclude it manually.
        """
        exclude: Set[str] = kwargs.get('exclude', set())
        if not isinstance(exclude, set):
            raise TypeError("exclude must be a set")
        exclude = exclude.union({'_batch_response_stats'})
        kwargs['exclude'] = exclude
        return super().dict(*args, **kwargs)

    def get_header_dict(self) -> Dict[str, str]:
        return self._batch_response_stats.get_header_dict()


class MarqoAddDocumentsItem(MarqoBaseModel):
    """A response from adding a document to Marqo.

    This model takes the response from Marqo vector store and translate it to a user-friendly response.
    """
    status: int
    # This id can be any type as it might be used to hold an invalid id response
    id: Any = Field(alias="_id", default=None)
    message: Optional[str] = None
    error: Optional[str] = None
    code: Optional[str] = None

    @classmethod
    def from_error(cls, doc_id: Optional[str], error: Any) -> 'MarqoAddDocumentsItem':
        return MarqoAddDocumentsItem(
            id=doc_id if doc_id is not None else '',
            error=error.message,
            message=error.message,
            status=int(error.status_code or 400),
            code=error.code
        )


class MarqoAddDocumentsResponse(MarqoBaseDocumentsResponse):
    errors: bool
    processingTimeMs: float
    index_name: str  # TODO Change this to camelCase in the future (Breaking change!)
    items: List[MarqoAddDocumentsItem]

    @root_validator(pre=False, skip_on_failure=True)
    def count_items(cls, values):
        items = values.get("items")
        batch_response_count = BatchResponseStats()

        if items:
            for item in items:
                if item.status in range(200, 300):
                    batch_response_count.success_count += 1
                elif item.status in range(400, 500):
                    batch_response_count.failure_count += 1
                elif item.status >= 500:
                    batch_response_count.error_count += 1
                else:
                    raise ValueError(f"Unexpected status code: {item.status}")

        values['_batch_response_stats'] = batch_response_count
        return values
