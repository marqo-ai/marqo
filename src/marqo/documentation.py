import marqo.version

version = marqo.version.__version__
base_url = 'https://docs.marqo.ai'


def _build_url(path):
    return f'{base_url}/{version}/{path}'


def configuring_marqo():
    return _build_url('Guides/Advanced-Usage/configuration/')


def create_index():
    return _build_url('API-Reference/Indexes/create_index/')
