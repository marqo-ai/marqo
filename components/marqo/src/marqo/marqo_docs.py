base_url = 'https://docs.marqo.ai'


def _build_url(path):
    return f'{base_url}/{path}'


# TODO: Update URL when a dedicated configuration page is available in the new docs
def configuring_marqo():
    return _build_url('reference/')


def create_index():
    return _build_url('reference/api/indexes/create-index/')


def multimodal_combination_object():
    return _build_url('reference/api/documents/add-or-replace-documents/#multimodal-combination')


def custom_vector_object():
    return _build_url('reference/api/documents/add-or-replace-documents/#custom-vectors')


def mappings():
    return _build_url('reference/api/documents/mappings/')


def map_fields():
    return _build_url('reference/api/documents/add-or-replace-documents/#map-fields')


# TODO: Update URL when the supported models page is fully migrated
def list_of_models():
    return _build_url('reference/')


def search_context():
    return _build_url('reference/api/search/#context')


# TODO: Update URL when a dedicated configuration page is available in the new docs
def configuring_preloaded_models():
    return _build_url('reference/')


# TODO: Update URL when a bring-your-own-model page is available in the new docs
def bring_your_own_model():
    return _build_url('reference/')


def query_reference():
    return _build_url('reference/api/search/#query-q')


# TODO: Update URL when a dedicated images/indexing guide is available in the new docs
def indexing_images():
    return _build_url('reference/')


def api_reference_document_body():
    return _build_url('reference/api/documents/add-or-replace-documents/#body')


# TODO: Update URL when a troubleshooting page is available in the new docs
def troubleshooting():
    return _build_url('reference/')


# TODO: Update URL when the supported models page is fully migrated
def generic_models():
    return _build_url('reference/')


def search_api_score_modifiers_parameter():
    return _build_url('reference/api/search/#score-modifiers')


# TODO: Update URL when a bring-your-own-model page is available in the new docs
def hugging_face_trust_remote_code():
    return _build_url('reference/')

def update_documents_response():
    return _build_url('reference/api/documents/update-documents/#response')

def hybrid_parameters():
    return _build_url('reference/api/search/#hybrid-parameters')
