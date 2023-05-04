"""Wrappers and helper functions for the s2_inference package and vectorise"""
from marqo.tensor_search.models.vectorise_params import VectoriseParams
from marqo.s2_inference.s2_inference import vectorise


def vectorise_from_params(vectorise_params: VectoriseParams, **kwargs):
    """A wrappper around the vectorise() function. This accepts the
    VectoriseParams class and propagates args through to vectorise()

    Any attributes set to None from vectorise_params gets discarded
    """
    return vectorise(vectorise_params.as_dict_discards_none(), **kwargs)
