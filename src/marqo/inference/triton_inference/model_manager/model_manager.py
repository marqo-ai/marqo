


class ModelManager:
    def __init__(self, triton_client):
        self.triton_client = triton_client
        pass

    def load_model(self, model_name: str):
        # Logic to load the model
        pass

    def eject_model(self, model_name: str):
        # Logic to unload the model
        pass

    def get_loaded_models(self):
        # Logic to get a list of loaded models
        pass
