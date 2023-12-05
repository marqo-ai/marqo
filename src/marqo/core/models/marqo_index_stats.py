from marqo.core.models.strict_base_model import StrictBaseModel


class MarqoIndexStats(StrictBaseModel):
    number_of_documents: int
    number_of_vectors: int
