"""
Our test suite is quite brittle. This helps the unit test suite navigate
refactoring transitions in Marqo
"""
from copy import deepcopy

from marqo.config import Config
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.tensor_search import add_documents
from marqo.tensor_search.utils import read_env_vars_and_defaults_ints


def add_docs_caller(config: Config, **kwargs):
    """This represents the call signature of add_documents at commit
    https://github.com/marqo-ai/marqo/commit/a884c840020e5f75b85b3d534b235a4a4b8f05b5

    New tests should NOT use this, and should call add_documents directly
    """

    # Add device = "cpu" to AddDocsParams if device not already specified in kwargs
    # add_documents can never be called without setting device first
    if "device" not in kwargs:
        kwargs["device"] = "cpu"
    
    return add_documents(config=config, add_docs_params=AddDocsParams(**kwargs))


def add_docs_batched(config: Config,
                     batch_size: int = read_env_vars_and_defaults_ints(EnvVars.MARQO_MAX_DOCUMENTS_BATCH_SIZE), **kwargs):
    """
    Helper function to batch large add_documents calls in testing
    Default batch size is the default max add docs count env var

    Must be called with raw kwargs (without AddDocsParams), as validation for this class will reject docs with large doc count.
    """
    docs = kwargs["docs"]
    kwargs_without_docs = deepcopy(kwargs)
    del kwargs_without_docs["docs"]

    for i in range(0, len(docs), batch_size):
        add_documents(
            config=config, add_docs_params=AddDocsParams(docs=docs[i:i+batch_size], **kwargs_without_docs)
        )