import httpx


class TritonModelManager:
    """
    A class to communicate with marqo model management container to load/eject models in Triton Inference Server.
    """

    def __init__(self, url: str):
        self.url = url
        self.client = httpx.Client()

    def load_model(self, model_properties: dict, timeout: float = 600):
        pay_load = {
            "tritonModelProperties": model_properties,
        }
        self.client.post(url=f"{self.url}/v1/models/load", json=pay_load, timeout=timeout)

    def eject_model(self, model_name: str, remove_files: bool = False, timeout: float = 60):
        self.client.post(url=f"{self.url}/v1/models/unload?remove-files={remove_files}", timeout=timeout)

    def get_loaded_models(self):
        raise NotImplementedError
