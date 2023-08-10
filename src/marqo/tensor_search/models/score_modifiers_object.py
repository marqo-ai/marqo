from typing import List, Dict, Any, Optional, Literal, Tuple
import json

from pydantic import BaseModel, validator, ValidationError

from marqo.errors import InvalidArgError


class ScoreModifierValidationError(InvalidArgError):
    def __init__(self, modifier: Dict[str, Any], message: str, link: str = None):
        super().__init__(
            link=link,
            message=f"Error validating score_modifiers = `{modifier}`. Reason: \n{message} "
                    f"Please revise your score_modifiers based on the provided error."
                    f"\n Check `https://docs.marqo.ai/0.0.17/API-Reference/search/#score-modifiers` for more info."
        )


class ScoreModifierOperator(BaseModel):
    field_name: str
    weight: float = 1

    class Config:
        extra: str = "forbid"
        allow_mutation = False

    @validator('field_name')
    def name_not_id(cls, v):
        if v == "_id":
            raise InvalidArgError("_id is not allowed as a field_name")
        return v

    def to_painless_script_and_params(self, field_index: int, operation: Literal["multiply_score_by", "add_to_score"]) \
            -> Tuple[str, Dict, Dict]:
        """Convert a ScoreModifierOperator based on if it's an multiply_score_by', 'add_to_score' operation.

           Returns: A tuple of (the painless script, the weights in the params, the field name in the params)
        """
        if operation == "multiply_score_by":
            return f"""
            if (doc.containsKey(params.multiplier_field_{field_index})) {{
                if (doc[params.multiplier_field_{field_index}].size() > 0 &&
                    (doc[params.multiplier_field_{field_index}].value instanceof java.lang.Number)) {{
                    _score = _score * doc[params.multiplier_field_{field_index}].value * params.multiplier_weight_{field_index};
                }}
            }}
            """, {f"multiplier_weight_{field_index}": self.weight}, {f"multiplier_field_{field_index}": f"__chunks.{self.field_name}"}

        elif operation == "add_to_score":
            return f""" 
            if (doc.containsKey(params.add_field_{field_index})) {{     
                if (doc[params.add_field_{field_index}].size() > 0 &&
                    (doc[params.add_field_{field_index}].value instanceof java.lang.Number)) {{
                    additive = additive + doc[params.add_field_{field_index}].value * params.add_weight_{field_index};
                }}
            }}
            """, {f"add_weight_{field_index}": self.weight}, {f"add_field_{field_index}": f"__chunks.{self.field_name}"}

        else:
            raise ValueError(
                f"operation must be either 'multiply_score_by' or 'add_to_score', {operation} provided."
            )


class ScoreModifier(BaseModel):
    multiply_score_by: Optional[List[ScoreModifierOperator]] = None
    add_to_score: Optional[List[ScoreModifierOperator]] = None

    class Config:
        extra: str = "forbid"

    def __init__(self, **data):
        try:
            super().__init__(**data)
        except ValidationError as e:
            raise ScoreModifierValidationError(modifier=data, message=json.dumps(e.errors()))

    @validator('multiply_score_by', 'add_to_score', pre=True, always=True)
    def at_least_one_must_be_provided(cls, v, values, field):
        """Validates that at least one of 'multiply_score_by', 'add_to_score' is non-null."""
        if field.name == 'add_to_score' and v is None and values.get('multiply_score_by') is None:
            raise ScoreModifierValidationError(modifier=values,
                                               message="At least one of multiply_score_by or add_to_score must be provided")
        return v

    @validator('multiply_score_by', 'add_to_score', pre=False, always=True)
    def at_least_one_item(cls, v, values, field):
        """Validate that, if present, the fields must be non-empty."""
        if v is not None and len(v) < 1:
            raise ScoreModifierValidationError(modifier=values,
                                               message=f"At least one ScoreModifierOperator is required in {field.name}")
        return v

    def to_script_score(self) -> dict:
        """Convert this ScoreModifier to a dictionary to modify the score.
        returns:
            A dictionary that can be used as a script_score in an OpenSearch query.
                source: the painless script
                params: the field name and fields used in the painless script
        """
        if self.multiply_score_by:
            mult, mult_weight_params, mult_field_params = map(list, zip(*([x.to_painless_script_and_params(field_index, "multiply_score_by") for field_index, x in
                                                 enumerate(self.multiply_score_by)])))
        else:
            mult, mult_weight_params, mult_field_params = [], [], []

        if self.add_to_score:
            add, add_weight_params, add_field_params = map(list, zip(*([x.to_painless_script_and_params(field_index, "add_to_score") for field_index, x
                                               in enumerate(self.add_to_score)])))
        else:
            add, add_weight_params, add_field_params = [], [], []

        source_script = "\n".join(
            ["double additive = 0;"] + \
            mult + add + ["return Math.max(0.0, (_score + additive));"]
        )

        params = {k: v for d in mult_weight_params + mult_field_params + add_weight_params + add_field_params for k, v in d.items()}
        script_score = {"source": f"""{source_script}""", "params": params}
        return script_score
