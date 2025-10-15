def validate_no_model(model_name: str, model_properties: dict):
    """
    A validation function to ensure that when using the "no_model" option.
    This model can't be used to generate embeddings.
    """
    model_type = model_properties.get("type", None)

    if model_type != "no_model" or model_name != "no_model":
        raise ValueError(
            f"To use no_model, type field in modelProperties must be 'no_model', and the name of the "
            f"model must be 'no_model'. Received type: {model_type}, name: {model_name}"
        )
