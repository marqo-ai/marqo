from pydantic.dataclasses import dataclass
from typing import Optional
from marqo.tensor_search.models.external_apis.abstract_classes import (
    ObjectLocation, ExternalAuth
)
from pydantic import validator, root_validator
from marqo.errors import InvalidArgError


class HfAuth(ExternalAuth):
    token: str


class HfModelLocation(ObjectLocation):
    repo_id: Optional[str]
    filename: Optional[str]
    name: Optional[str]
    @root_validator(pre=True)
    def validate_input(cls, values):
        repo_id = values.get('repo_id')
        filename = values.get('filename')
        name = values.get('name')

        if (repo_id is not None and filename is None) or (repo_id is None and filename is not None):
            raise InvalidArgError(
                "In `type = hf` model properties, 'repo_id' and 'filename' should be provided together.")

        if name is not None and (repo_id is not None or filename is not None):
            raise InvalidArgError(
                "In `type = hf` model properties, if 'name' is provided, 'repo_id' and 'filename' should not be provided.")

        if name is None and (repo_id is None or filename is None):
            raise InvalidArgError(
                "In `type = hf` model properties, either 'name' or both 'repo_id' and 'filename' should be provided.")

        return values



