from marqo.base_model import ImmutableStrictBaseModel


class RollbackRequest(ImmutableStrictBaseModel):
    from_version: str
    to_version: str
