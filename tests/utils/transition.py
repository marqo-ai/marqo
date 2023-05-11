"""
Our test suite is quite brittle. This helps the unit test suite navigate
refactoring transitions in Marqo
"""
from marqo.tensor_search.tensor_search import add_documents
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.config import Config

def add_docs_caller(config: Config, **kwargs):
    """This represents the call signature of add_documents at commit
    https://github.com/marqo-ai/marqo/commit/a884c840020e5f75b85b3d534b235a4a4b8f05b5

    New tests should NOT use this, and should call add_docs directly
    """
    return add_documents(config=config, add_docs_params=AddDocsParams(**kwargs))