from typing import Union, Optional, Tuple, Dict, Sequence, List, Any, Set
from marqo.tensor_search.enums import MediaType
from marqo import errors
from marqo.s2_inference.s2_inference import vectorise
from marqo.tensor_search import utils


class VectoriseArgs:
    """Class that holds args to be executed by vectorise()
    We might need to implement hash and or equality
    """

    @staticmethod
    def validate_content_type(content_type):
        valid_content_types = [MediaType.text, MediaType.image]
        if content_type not in valid_content_types:
            raise ValueError(
                f"VectoriseArgs got unknown content_type: `{content_type}` "
                f"content_type must be one of `{valid_content_types}`")
        return content_type

    def __init__(self, content_type: Union[MediaType.text, MediaType.image, str], **vec_kwargs):
        self.vec_kwargs = vec_kwargs
        self.content_type = self.validate_content_type(content_type)


# class ComboVectoriseArgs:
#     # TODO, after first iteration
#     """
#         This needs to have this structure: {
#             args = [(content, VectoriseArgs, PostVectoriseArgs)]
#             combination_args = {}
#         }
#
#     """
#
#     def __init__(self, ):

class Content4Vectorising:
    """Represents a single piece of content to vectorise.

    This represents objects post-chunking.
    """
    def __init__(self, string_representation: str, content_to_vectorise: Any = None):
        """

        Args:
            string_representation: The string representation of the data to be
                vectorised. For text content, this would be the same as
                content_to_vectorise. For images, this could be the URL to the
                image.
            content_to_vectorise: If this is not given, string_representation
                is used instead (such as when the content to be vectorised is
                text data). For images, this can be a PIL image
        """
        self.string_representation = string_representation
        if content_to_vectorise is None:
            self.content_to_vectorise = string_representation
        else:
            self.content_to_vectorise = content_to_vectorise

    def __hash__(self):
        return hash(self.string_representation)

    def __repr__(self):
        return (f"<{self.__class__.__name__} string_representation: {self.string_representation}, "
                f"content_to_vectorise: {self.content_to_vectorise}>")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.string_representation == other.string_representation


def execute_bulk_vectorise(
        to_be_vectorised_dict: Dict[VectoriseArgs, Set[Content4Vectorising]]
) -> Dict[VectoriseArgs, Dict[Content4Vectorising, Sequence[float]]]:
    """Vectorises sets of content that share the same vectorise arguments.

    From a mapping of <VectoriseArguments: Set(content to vectorise)> a mapping
    <VectoriseArguments: Dict<content: vector>> is created.

    Returns a new dict with the structure <VectoriseArguments: Dict<content: vector>>
    """
    args2content_vectors = dict()
    for vectorise_arg in to_be_vectorised_dict:
        content_as_list = list(to_be_vectorised_dict[vectorise_arg])
        content2vectors = dict(zip(
            content_as_list,
            vectorise(**vectorise_arg.vec_kwargs, content=[
                content.content_to_vectorise for content in content_as_list])
        ))
        args2content_vectors[vectorise_arg] = content2vectors
    return args2content_vectors


def distribute_vectors_to_chunks(
        docs: List[Dict], vectorised_dict: Dict[VectoriseArgs, Dict[Content4Vectorising, Sequence[float]]]) -> None:
    """ Takes the bulk parent dicts, and distributes the docs to them
    TODO:
        - use enum instead of __chunks
        - consider ranming bulk_parent_dicts
        - We need to ensure we separate the modalities (include tests)
    Args:
        docs:
        vectorised_dict:

    Returns:

    """
    for doc in docs:
        if '__chunks' not in doc:
            # also skips the indexing instructions
            continue
        for chunk in doc['__chunks']:
            field_name = chunk['__field_name']
            knn_field_name = utils.generate_vector_name(field_name)
            vectorise_args_hash = chunk[knn_field_name]
            # get the content2vec dict for this set of args
            content2vec = vectorised_dict[vectorise_args_hash]

            chunk[knn_field_name] = content2vec[Content4Vectorising(string_representation=chunk['__field_content'])]

