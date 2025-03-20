from typing import Optional

import numpy as np

from marqo.core.inference.api import Inference, InferenceRequest, InferenceResult, InferenceError

# TODO move device manager to native_inference
from marqo.core.inference.device_manager import DeviceManager
from marqo.inference.native_inference.local_inference import NativeInferenceLocal


class Config:
    def __init__(
            self,
    ) -> None:

        # TODO [Refactoring device logic] deprecate default_device since it's not used
        # self.default_device = default_device if default_device is not None else (
        #     utils.read_env_vars_and_defaults(EnvVars.MARQO_BEST_AVAILABLE_DEVICE))

        # TODO load env vars to this class and expose them as properties

        self.device_manager: DeviceManager = DeviceManager()
        self.local_inference: Inference = NativeInferenceLocal(device_manager=self.device_manager)

