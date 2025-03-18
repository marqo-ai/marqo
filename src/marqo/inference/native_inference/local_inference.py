from marqo.inference.native_inference.inference_pipeline import InferencePipeline
from marqo.inference.type import *



class NativeInferenceLocal(Inference):

    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        return InferencePipeline(request).run_pipeline()
