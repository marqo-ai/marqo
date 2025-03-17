from marqo.inference.native_inference.chunk_download_preprocess_content import chunk_download_preprocess_content
from marqo.inference.native_inference.embedding_models.hugging_face_model import HuggingFaceModel
from marqo.inference.native_inference.embedding_models.languagebind_model import LanguagebindModel
from marqo.inference.native_inference.embedding_models.open_clip_model import OPEN_CLIP
from marqo.inference.native_inference.encode_content import encode_processed_content, format_results
from marqo.inference.native_inference.load_model import load_model
from marqo.inference.type import *


class InferencePipeline:
    def __init__(self, inference_request: InferenceRequest):
        self.inference_request = inference_request

    def run_pipeline(self) -> InferenceResult:
        model = load_model(
            model_name=self.inference_request.model_config.model_name,
            model_properties=self.inference_request.model_config.model_properties,
            model_auth=self.inference_request.model_config.model_auth,
            device=self.inference_request.device
        )

        if isinstance(model, OPEN_CLIP):
            return self._run_open_clip_inference_pipeline(model)
        elif isinstance(model, HuggingFaceModel):
            return self._run_hf_inference_pipeline(model, self.inference_request)
        elif isinstance(model, LanguagebindModel):
            return self._run_languagebind_inference_pipeline(model, self.inference_request)
        else:
            raise ValueError(f"Model type {type(model)} not supported")


    def _run_open_clip_inference_pipeline(self, model) -> InferenceResult:
        # Chunk, download, and preprocess the content
        preprocessed_content_list: List[PreprocessedContent] = chunk_download_preprocess_content(
            content=self.inference_request.contents,
            modality=self.inference_request.modality,
            preprocessing_config=self.inference_request.preprocessing_config,
            preprocessor=model.get_preprocessor(),
        )

        embeddings: List[Tensor] = encode_processed_content(
            model=model,
            preprocessed_content_list=preprocessed_content_list,
            modality=self.inference_request.modality,
            normalize=self.inference_request.model_config.normalize_embeddings,
            max_batch_size=16
        )

        # Format the results
        formated_result = format_results(preprocessed_content_list, embeddings)
        return formated_result

    def _run_languagebind_inference_pipeline(self, model, inference_request) -> InferenceResult:
        pass

    def _run_hf_inference_pipeline(self, model, inference_request) -> InferenceResult:
        pass