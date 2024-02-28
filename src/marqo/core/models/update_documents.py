from typing import Dict, Sequence, Any

from marqo.base_model import ImmutableStrictBaseModel


class UpdateDocumentsBodyParams(ImmutableStrictBaseModel):
    documents: Sequence[Dict[str, Any]]

