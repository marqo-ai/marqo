from pydantic.dataclasses import dataclass
from typing import Optional
from marqo.tensor_search.models.external_apis.abstract_classes import ExternalAuth, ObjectLocation


@dataclass(frozen=True)
class VectoriseParams:
    """Holds params to vectorise, besides the content to be vectorised

    Don't include any attributes that start with '__' as these are discarded by
    as_dict_discards_none()
    """
    model_name: Optional[str] = None
    model_properties: Optional[dict] = None
    device: Optional[str] = None
    normalize_embeddings: Optional[bool] = None
    model_auth: Optional[ExternalAuth] = None
    model_location: Optional[ObjectLocation] = None

    def as_dict_discards_none(self):
        """Returns a dict version of the object's attributes, discarding any k,v
        pairs where the value is None.

        Discards attributes that start with '__'
        """
        return {k: v for k, v in self.__dict__.items() if v is not None and not k.startswith('__')}




