from typing import Union, List, Tuple

from numpy import ndarray

from marqo.base_model import ImmutableBaseModel
from marqo.core.inference.api import InferenceError, InferenceResult


class InferenceErrorModel(ImmutableBaseModel):
    error_code: str
    message: str

    @classmethod
    def from_inference_error(cls, error: InferenceError) -> 'InferenceErrorModel':
        # factory method to create InferenceErrorModel out of InferenceError
        pass

    def to_inference_error(self) -> InferenceError:
        pass


class InferenceResponse(ImmutableBaseModel):
    result: List[Union[InferenceErrorModel, List[Tuple[str, ndarray]]]]

    class Config(ImmutableBaseModel.Config):
        arbitrary_types_allowed = True

    @classmethod
    def from_inference_result(cls, result: InferenceResult) -> 'InferenceResponse':
        return InferenceResponse(result=
                                 [InferenceErrorModel.from_inference_error(r) if isinstance(r, InferenceError) else r
                                  for r in result.result])

    def to_inference_result(self) -> InferenceResult:
        return InferenceResult(result=[r.to_inference_error() if isinstance(r, InferenceErrorModel) else r
                                       for r in self.result])