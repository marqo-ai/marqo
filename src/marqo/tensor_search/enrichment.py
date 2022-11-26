from marqo.s2_inference.s2_inference import generate
from typing import List
"""
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
        kwargs = _parse_kwargs(enrichment["kwargs"], doc)
        # Pandu - we may might want to call this task?  enrichment["model"] -> enrichment["task"]
        generated = generate(enrichment["model"], device, [], kwargs)
        for i, to_field in enumerate(enrichment["to"]):
            doc[to_field] = generated[i]
    return documents


def _parse_kwargs(doc: dict, kwargs: dict) -> dict:
    """
     {
            "attributes": [{"string": "Bathroom, Bedroom, Study, Yard"} ]
            "image_field": {'document_field":"Image location"}
     	},
     	->
    {
            "attributes": ["Bathroom, Bedroom, Study, Yard"]
            "image_field": "https://s3.image.png"
     	},
    The names of keyword arguments are fixed according to the model. The model should validate
    the kwargs
    TODO
    """
    return dict()