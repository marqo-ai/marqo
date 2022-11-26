from marqo.s2_inference.s2_inference import generate
from typing import List
from marqo import errors
from marqo.tensor_search import constants
"""
Example: 
enrich( documents=[{
        "Image Location": "https://s3.image.png",
       "Description": "A photo of a house" 
   }],
  enrichment={
    "model": "attribute-extraction,
    "to": ["Is_bathroom", "Is_Bedroom", Is_Study", "Is_Yard"], 
    "kwargs" : {
        "attributes": [{"string": "Bathroom, Bedroom, Study, Yard"} ]
        "image_field": {'document_field":"Image location"}
    },
    },
)

"""


def enrich(documents: List[dict], enrichment: dict, device, indexing_instructions=None):
    for doc in documents:
        kwargs = _parse_kwargs(doc=doc, kwargs=enrichment["kwargs"])
        generated = generate(task=enrichment["task"], device=device, **kwargs)
        for i, to_field in enumerate(enrichment["to"]):
            doc[to_field] = generated[i]
    return {
        "documents": documents,
        "results": [
            {"result": "successful"} for _ in documents
        ]
    }


def _parse_kwargs(doc: dict, kwargs: dict) -> dict:
    """
    The names of keyword arguments are fixed according to the model. The model should validate
    the kwargs.

    Returns:
        Parsed expanded keyword arg dicts:
        {
            "attributes": [{"string": "Bathroom, Bedroom, Study, Yard"} ]
            "image_field": {'document_field":"Image location"}
        },
        ->
        {
            "attributes": ["Bathroom, Bedroom, Study, Yard"]
            "image_field": "https://s3.image.png"
        }

    Raises:
        errors.ArgParseError
    """
    parsed = dict()
    for keyword, argument in kwargs.items():
        if isinstance(argument, List):
            parsed[keyword] = [_expand_arg(doc=doc, arg=kwarg) for kwarg in argument]
        elif isinstance(argument, dict):
            parsed[keyword] = _expand_arg(doc=doc, arg=argument)
        else:
            raise errors.ArgParseError(
                f"Keyword argument must be a list or object! "
                f"Received keyword argument `{argument}`. \n"
                'Example of valid keyword arguments: {"string": "What color are plants?"} and  '
                '{"document_field": "Title"}. \n')
    return parsed


def _expand_arg(doc: dict, arg: dict) -> str:
    """ Expands the arg, based on the parse instruction

    Args:
        doc: The document associated with this parse
        arg: A len-1 dict, with the key as the arg type
            e.g: 'string' or 'document_field' and the value being the argument
            content. The argument content gets expanded in respect to the
            content type.
            For example:
                {'document_field":"Image location"}
                {"string": "Bathroom, Bedroom, Study, Yard"}

    Returns:
         A string of the expanded content:
            {"string": "Bathroom, Bedroom, Study, Yard"}
            -> "Bathroom, Bedroom, Study, Yard"
    Raises:
        errors.ArgParseError
    """
    print('argarg', arg)
    if not isinstance(arg, dict):
        raise errors.ArgParseError(
            f"Keyword argument object must be a dictionary! "
            f"Received keyword argument `{arg}` of type {type(arg).__name__}. \n"
            'Example of valid keyword arguments: {"string": "What color are plants?"} and  '
            '{"document_field": "Title"}. \n'
        )
    if len(arg) != 1:
        raise errors.ArgParseError(
            f"Keyword argument object can only have a single field! "
            f"Received keyword argument `{arg}`. \n"
            'Example of valid keyword arguments: {"string": "What color are plants?"} and  '
            '{"document_field": "Title"}. \n'
        )
    parse_instruction = list(arg.keys())[0]

    if parse_instruction not in constants.VALID_ENRICHMENT_KWARG_PARSE_INSTRUCTIONS:
        raise errors.ArgParseError(
            f"Keyword argument object had unknown key! "
            f"Received keyword argument `{arg}`. \n"
            f'Key must be one of {constants.VALID_ENRICHMENT_KWARG_PARSE_INSTRUCTIONS} \n'
        )

    if parse_instruction == 'string':
        return arg[parse_instruction]
    elif parse_instruction =='document_field':
        return doc[arg[parse_instruction]]
    else:
        raise ValueError(f'unhandled parse instruction {parse_instruction}')



