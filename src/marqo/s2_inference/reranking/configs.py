from marqo.core.models.marqo_index import TextPreProcessing
from marqo.core.models.marqo_index import TextSplitMethod

def get_default_text_processing_parameters() -> TextPreProcessing:
    return TextPreProcessing(
        splitLength=2,
        splitOverlap=0,
        splitMethod=TextSplitMethod.Sentence
    )