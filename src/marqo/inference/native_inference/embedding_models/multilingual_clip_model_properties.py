from marqo.inference.native_inference.embedding_models.marqo_base_model_properties import MarqoBaseModelProperties


class MultilingualCLIPModelProperties(MarqoBaseModelProperties):
    """
    A class to represent the properties of an OpenCLIP model.

    Attributes:
        name: The name of the model. It can be the name of the model for loading information. e.g., the
            architecture name of the model, the name of the model in the Hugging Face model hub, etc. It might be
            the same as the model tag but this is not necessary.
        type: The type of the model. It should be 'multilingual_clip'.
        visual_model: The name of the visual model.
        textual_model: The name of the textual model.
        dimensions: The dimensions of the model.

    """
    name: str
    visual_model: str
    textual_model: str
