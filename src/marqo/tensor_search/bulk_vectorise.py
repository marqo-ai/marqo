from typing import Union, Optional, Tuple, Dict
from marqo.tensor_search.enums import MediaType
from marqo import errors
from marqo.s2_inference.s2_inference import vectorise

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


def execute_bulk_vectorise(to_be_vectorised_dict: Dict[VectoriseArgs, Tuple[set, Optional[dict]]]) -> None:
    """"""
    for vectorise_arg in to_be_vectorised_dict:
        content_as_list = list(to_be_vectorised_dict[vectorise_arg][0])
        content2vectors = dict(zip(
            content_as_list,
            vectorise(**vectorise_arg.vec_kwargs, content=content_as_list)
        ))
        to_be_vectorised_dict[vectorise_arg] = (
            to_be_vectorised_dict[vectorise_arg][0],
            content2vectors
        )
