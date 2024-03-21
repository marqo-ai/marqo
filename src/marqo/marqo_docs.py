import marqo.version

version = marqo.version.get_version()
base_url = 'https://docs.marqo.ai'


def _build_url(path):
    return f'{base_url}/{version}/{path}'


def configuring_marqo():
    return _build_url('Guides/Advanced-Usage/configuration/')


def create_index():
    return _build_url('API-Reference/Indexes/create_index/')


def multimodal_combination_object():
    return _build_url('Guides/Advanced-Usage/document_fields/#multimodal-combination-object/')


def custom_vector_object():
    return _build_url('Guides/Advanced-Usage/document_fields/#custom-vectors/')


def mappings():
    return _build_url('API-Reference/Documents/mappings/')


def list_of_models():
    return _build_url('Guides/Models-Reference/list_of_models/')


def search_context():
    return _build_url('API-Reference/Search/search/#context')


def configuring_preloaded_models():
    return _build_url('Guides/Advanced-Usage/configuration/#configuring-preloaded-models')

