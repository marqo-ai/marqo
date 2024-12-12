from typing import Optional, List

from pydantic import Field, root_validator
from pydantic import validator

from marqo.base_model import MarqoBaseModel
from marqo.core.inference.embedding_models.marqo_base_model_properties import MarqoBaseModelProperties
from marqo.s2_inference.types import Modality
from marqo.tensor_search.models.external_apis.hf import HfModelLocation
from marqo.tensor_search.models.external_apis.s3 import S3Location


class ModalityLocation(MarqoBaseModel):
    """Location of the Modality.

    This stores the location of the model for each modality.
    """
    s3: Optional[S3Location] = None
    hf: Optional[HfModelLocation] = None
    authRequired: bool = Field(default=False, alias="auth_required")
    url: Optional[str] = None

    @root_validator(skip_on_failure=True)
    def _validate_minimum_provided_fields(cls, values):
        """Validate that at least one location is provided."""
        s3 = values.get("s3")
        hf = values.get("hf")
        url = values.get("url")
        if sum([1 for x in [s3, hf, url] if x]) != 1:
            raise ValueError("Exactly one of url, s3, hf must be provided to load the model")
        return values

    @root_validator(skip_on_failure=True)
    def _validate_auth_required(cls, values):
        """Validate that authRequired can only be set to True when s3 or hf is provided."""
        auth_required = values.get("authRequired")
        url = values.get("url")
        if url and auth_required:
            raise ValueError("authRequired must be False when url is provided. It only works with s3 or hf")


class LanguagebindModelLocation(MarqoBaseModel):
    """Location of the LanguagebindModel.

    This is a wrapper class for the location of each modality.
    """
    audio: Optional[ModalityLocation] = None
    image: Optional[ModalityLocation] = None
    video: Optional[ModalityLocation] = None
    tokenizer: Optional[ModalityLocation] = None

    @root_validator(skip_on_failure=True)
    def _validate_minimum_provided_fields(cls, values):
        """Validate that at least one location is provided."""
        audio = values.get("audio")
        image = values.get("image")
        video = values.get("video")

        if sum([1 for x in [audio, image, video] if x]) == 0:
            raise ValueError("At least one of audio, image, video must be provided to load the model")


class LanguagebindModelProperties(MarqoBaseModelProperties):
    """Properties of the LanguagebindModel class.

    name: The name of the model. Only used when loading Marqo registered models.
    modelLocation: The location of the model for each modality.
    supportedModalities: The supported modalities of the model.
    """
    name: Optional[str]
    modelLocation: Optional[LanguagebindModelLocation]
    supportedModalities: List[Modality] = Field(alias="supported_modalities")

    @validator('type')
    def _type_must_be_languagebind(cls, v):
        if v != "languagebind":
            raise ValueError('type must be "languagebind" for this model')

    @root_validator(pre=False, skip_on_failure=True)
    def _validate_minimum_required_fields_to_load_model(cls, values):
        """Validate the minimum required fields to load the model."""
        name = values.get("name")
        model_location = values.get("modelLocation")
        if not name and not model_location:
            raise ValueError("Either name or modelLocation must be provided to load the model")
        elif name and model_location:
            raise ValueError("Only one of name or modelLocation must be provided to load the model")
        return values

    @root_validator(pre=False, skip_on_failure=True)
    def _validate_modalities_match_model_location(cls, values):
        """Validate that the supported modalities match the model location."""
        model_location = values.get("modelLocation")
        if model_location:
            for modality in values.get("supportedModalities"):
                if modality == Modality.TEXT:
                    # Skip the check for text modality
                    continue
                if modality not in model_location.dict().keys():
                    raise ValueError(f"The supported modality '{modality}' is not in the model location")
        else:
            return values
