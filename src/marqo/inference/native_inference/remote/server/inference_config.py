from marqo.core.inference.api import Inference, ModelManager
from marqo.inference.native_inference.device_manager import DeviceManager
from marqo.inference.native_inference.load_model import NativeModelManager

# TODO move device manager to native_inference
from marqo.inference.native_inference.local_inference import NativeInferenceLocal


class Config:
    def __init__(self):
        # TODO load env vars to this class and expose them as properties
        self.device_manager: DeviceManager = DeviceManager()
        self.local_inference: Inference = NativeInferenceLocal(device_manager=self.device_manager)
        self.model_manager: ModelManager = NativeModelManager()

