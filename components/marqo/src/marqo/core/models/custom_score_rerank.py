"""Parsing of custom score rerank keys (suffix after ``marqo__score_``)."""
from __future__ import annotations

from typing import Literal, Optional, cast, get_args

from marqo.base_model import StrictBaseModel

ScoreType = Literal["bm25", "closeness_retrieval_vector"]
AggregateType = Literal["sum", "max", "avg"]



class ParsedCustomScoreKey(StrictBaseModel):
    """
    Parsed custom score rerank key.

    ``key`` is the suffix after the ``marqo__score_`` prefix, e.g. ``bm25_field_title``,
    not ``marqo__score_bm25_field_title``.

    Use :meth:`parse` to build from a string; invalid keys return ``None``.
    """

    score_type: ScoreType
    field_name: Optional[str] = None
    aggregate_type: Optional[AggregateType] = None

    @classmethod
    def parse(cls, key: str) -> Optional[ParsedCustomScoreKey]:
        """
        Parse `key` (suffix after `marqo__score_`) into the model fields.

        **Per-field** — BM25 on one lexical field, or vector closeness on one tensor field::

            bm25_field_title                      
                → score_type=bm25, field_name=title, aggregate_type=None
            closeness_retrieval_vector_field_img 
                → score_type=closeness_retrieval_vector, field_name=img, aggregate_type=None

        **Aggregate** — sum / max / avg over all lexical or all tensor fields::

            bm25_sum                       
                → score_type=bm25, field_name=None, aggregate_type=sum
            closeness_retrieval_vector_max 
                → score_type=closeness_retrieval_vector, field_name=None, aggregate_type=max

        Invalid keys return `None`.
        """
        if not key or "_" not in key:
            return None

        for score_type in get_args(ScoreType):
            prefix = score_type + "_"
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            if rest in get_args(AggregateType):
                # Set `aggregate_type` and leave `field_name` as None.
                return cls(
                    score_type=cast(ScoreType, score_type),
                    field_name=None,
                    aggregate_type=cast(AggregateType, rest),
                )
            if rest.startswith("field_"):
                field_name = rest.removeprefix("field_")
                if not field_name:
                    return None
                # Set `field_name` and leave `aggregate_type` as None.
                return cls(
                    score_type=cast(ScoreType, score_type),
                    field_name=field_name,
                    aggregate_type=None,
                )
            return None

        return None
