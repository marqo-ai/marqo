from typing import Optional, List

from pydantic import Field, root_validator
from pydantic import validator

from marqo.base_model import MarqoBaseModel
from marqo.inference.native_inference.embedding_models.marqo_base_model_properties import MarqoBaseModelProperties
from marqo.s2_inference.types import Modality
from marqo.tensor_search.models.external_apis.hf import HfModelLocation
from marqo.tensor_search.models.external_apis.s3 import S3Location


class ModalityLocation(MarqoBaseModel):
    """Location of the Modality.

    This stores the location of the model for each modality.
    """
    s3: Optional[S3Location] = None
    hf: Optional[HfModelLocation] = None
    url: Optional[str] = None

    @root_validator(skip_on_failure=True)
    def _validate_minimum_provided_fields(cls, values):
        """Validate that at exactly one location is provided."""
        s3 = values.get("s3")
        hf = values.get("hf")
        url = values.get("url")
        if sum([1 for x in [s3, hf, url] if x]) != 1:
            raise ValueError("Exactly one of url, s3, hf must be provided to load the model")
        return values


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
        return values


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
        return v

    @validator('supportedModalities', pre=True)
    def _validate_supported_modalities_text_must_be_supported(cls, v):
        """
        Validate that the supported modalities include 'text' or 'language'.
        'language' is deprecated in the API usage and should be replaced with 'text'.

        Raises:
            ValueError: If 'text' or 'language' is not in the supported modalities.
            ValueError: If both 'text' and 'language' are in the supported modalities.

        Returns:
            Return the supported modalities if either 'text' or 'language' is in the supported modalities.
        """
        if not isinstance(v, list):
            raise ValueError("Invalid data type, must be a list")
        if Modality.TEXT not in v and "text" not in v:
            raise ValueError("You model must include 'text' as a supported modality")
        if Modality.TEXT in v and "text" in v:
            raise ValueError("You cannot have both 'text' and 'language' as supported modalities. 'language' is "
                             "deprecated. Please use 'text' instead")
        # Replace 'text' with 'language' as we still use 'language' internally
        v = list(set([Modality.TEXT if x == "text" else x for x in v]))
        return v

    @validator('supportedModalities')
    def _validate_supported_modalities_minimum_supported(cls, v):
        """
        At least one of video, image, audio must be supported.
        """
        if Modality.VIDEO not in v and Modality.IMAGE not in v and Modality.AUDIO not in v:
            raise ValueError("At least one of 'video', 'image', 'audio' must be supported")
        return v

    @root_validator(pre=False, skip_on_failure=True)
    def _validate_minimum_required_fields_to_load_model(cls, values):
        """Validate the minimum required fields to load the model.

        Either name or modelLocation must be provided to load the model.
        If name is provided, this is a Marqo registered model.
        If modelLocation is provided, this is a custom model.

        Raises:
            ValueError: If neither name nor modelLocation is provided.
            ValueError: If both name and modelLocation are provided.
        """
        name = values.get("name")
        model_location = values.get("modelLocation")
        if not name and not model_location:
            raise ValueError("Either name or modelLocation must be provided to load the model")
        elif name and model_location:
            raise ValueError("Only one of name or modelLocation must be provided to load the model")
        return values

    @root_validator(pre=False, skip_on_failure=True)
    def _validate_modalities_match_model_location(cls, values):
        """Validate that the supported modalities match the model location. We skip the check for text and tokenizers
        as they are not required.

        Each modality must have a corresponding location in the model location, except for text and language modalities.
        """
        model_location = values.get("modelLocation")
        if model_location is not None:
            supported_modalities = values.get("supportedModalities")
            for supported_modality in supported_modalities:
                if supported_modality not in [Modality.TEXT, "test"]: # Skip text
                    if not getattr(model_location, supported_modality.lower()):
                        raise ValueError(f"Mismatch between supported modalities and model location. The supported "
                                         f"modality {supported_modality} does not have a corresponding modelLocation "
                                         f"in the model")
        return values