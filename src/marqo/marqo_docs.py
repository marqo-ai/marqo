import semver

import marqo.version

version = marqo.version.get_version()
base_url = 'https://docs.marqo.ai'

parsed_version = semver.VersionInfo.parse(version, optional_minor_and_patch=True)
docs_version = f'{parsed_version.major}.{parsed_version.minor}'


def _build_url(path):
    return f'{base_url}/{docs_version}/{path}'


def configuring_marqo():
    return _build_url('other-resources/guides/advanced-usage/configuration/')


def create_index():
    return _build_url('reference/api/indexes/create-index/')


def multimodal_combination_object():
    return _build_url('other-resources/guides/advanced-usage/document-fields/#multimodal-combination-object')


def custom_vector_object():
    return _build_url('other-resources/guides/advanced-usage/document-fields/#custom-vector-object')


def mappings():
    return _build_url('reference/api/documents/mappings/')


def map_fields():
    return _build_url('reference/api/documents/add-or-replace-documents/#map-fields')


def list_of_models():
    return _build_url('models/marqo/list-of-models/')


def search_context():
    return _build_url('reference/api/search/search/#context')


def configuring_preloaded_models():
    return _build_url('other-resources/guides/advanced-usage/configuration/#configuring-preloaded-models')


def bring_your_own_model():
    return _build_url('models/marqo/bring-your-own-model')


def query_reference():
    return _build_url('reference/api/search/search/#query-q')


def indexing_images():
    return _build_url('other-resources/guides/advanced-usage/images/')


def api_reference_document_body():
    return _build_url('reference/api/documents/add-or-replace-documents/#body')


def troubleshooting():
    return _build_url('other-resources/troubleshooting/troubleshooting/')


def generic_models():
    return _build_url('models/marqo/list-of-models/#generic-clip-models')


def search_api_score_modifiers_parameter():
    return _build_url('reference/api/search/search/#score-modifiers')


def hugging_face_trust_remote_code():
    return _build_url('models/marqo/bring-your-own-model/#bring-your-own-hugging-face-sentence-transformers-models')

def update_documents_response():
    return _build_url('reference/api/documents/update-documents/#response')
