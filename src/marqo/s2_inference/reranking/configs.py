from marqo.s2_inference.types import Dict

def get_default_text_processing_parameters() -> Dict:
    return {"split_length": 2, "split_overlap": 0, "split_method": "sentence"}