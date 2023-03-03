import json
from pydantic import BaseModel
from typing import Any, Union, List, Dict, Optional, NewType

Qidx = NewType('Qidx', int)
JHash = NewType('JHash', int)

class VectorisedJobPointer(BaseModel):
    job_hash: JHash
    start_idx: int
    end_idx: int

class VectorisedJobs(BaseModel):
    model_name: str
    model_properties: Dict[str, Any]
    content: List[Union[str, List[str]]]
    device: str
    normalize_embeddings: bool
    image_download_headers: Optional[Dict]

    def __hash__(self):
        return self.groupby_key() + hash(json.dumps(self.content, sort_keys=True))

    def groupby_key(self) -> JHash:
        return VectorisedJobs.get_groupby_key(self.model_name, self.model_properties, self.device, self.normalize_embeddings, self.image_download_headers)

    @staticmethod
    def get_groupby_key(model_name: str, model_properties: Dict[str, Any], device: str,
                        normalize_embeddings: bool, image_download_headers: Optional[Dict]) -> JHash:
        return JHash(hash(model_name) + hash(json.dumps(model_properties, sort_keys=True))
                     + hash(device) + hash(normalize_embeddings)
                     + hash(json.dumps(image_download_headers, sort_keys=True)))

    def add_content(self, content: List[Union[str, List[str]]]) -> VectorisedJobPointer:
        start_idx = len(self.content)
        self.content.extend(content)

        return VectorisedJobPointer(
            job_hash=self.groupby_key(),
            start_idx=start_idx,
            end_idx=len(self.content)
        )
