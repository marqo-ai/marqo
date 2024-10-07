import semver

import marqo.version

version = marqo.version.get_version()
base_url = 'https://docs.marqo.ai'

parsed_version = semver.VersionInfo.parse(version, optional_minor_and_patch=True)
docs_version = f'{parsed_version.major}.{parsed_version.minor}'


def _build_url(path):
    return f'{base_url}/{docs_version}/{path}'


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


def map_fields():
    return _build_url('API-Reference/Documents/add_or_replace_documents/#map-fields/')


def list_of_models():
    return _build_url('Guides/Models-Reference/list_of_models/')


def search_context():
    return _build_url('API-Reference/Search/search/#context')


def configuring_preloaded_models():
    return _build_url('Guides/Advanced-Usage/configuration/#configuring-preloaded-models')


def bring_your_own_model():
    return _build_url('Guides/Models-Reference/bring_your_own_model/')


def query_reference():
    return _build_url('API-Reference/Search/search/#query-q')


def indexing_images():
    return _build_url('Guides/Advanced-Usage/images/')


def api_reference_document_body():
    return _build_url('API-Reference/Documents/add_or_replace_documents/#body')


def troubleshooting():
    return _build_url('other-resources/troubleshooting/troubleshooting/')
