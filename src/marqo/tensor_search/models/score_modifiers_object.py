from typing import List, Dict, Any, Optional, Literal
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

    @validator('field_name')
    def name_not_id(cls, v):
        if v == "_id":
            raise InvalidArgError("_id is not allowed as a field_name")
        return v

    def to_painless_script(self, operation: Literal["multiply_score_by", "add_to_score"]) -> str:
        """Convert a ScoreModifierOperator based on if it's an multiply_score_by', 'add_to_score' operation."""
        if operation == "multiply_score_by":
            return f"""
            if (doc.containsKey('__chunks.{self.field_name}')) {{
                if (doc['__chunks.{self.field_name}'].size() > 0 &&
                    (doc['__chunks.{self.field_name}'].value instanceof java.lang.Number)) {{
                    _score = _score * doc['__chunks.{self.field_name}'].value * {self.weight};
                }}
            }}
            """

        elif operation == "add_to_score":
            return f""" 
            if (doc.containsKey('__chunks.{self.field_name}')) {{     
                if (doc['__chunks.{self.field_name}'].size() > 0 &&
                    (doc['__chunks.{self.field_name}'].value instanceof java.lang.Number)) {{
                    additive = additive + doc['__chunks.{self.field_name}'].value * {self.weight};
                }}
            }}
            """

        else:
            raise ValueError(
                f"operation must be either 'multiplicative' or 'additive', {operation} provided."
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
            raise ScoreModifierValidationError(modifier=values, message="At least one of multiply_score_by or add_to_score must be provided")
        return v

    @validator('multiply_score_by', 'add_to_score', pre=False, always=True)
    def at_least_one_item(cls, v, values, field):
        """Validate that, if present, the fields must be non-empty."""
        if v is not None and len(v) < 1:
            raise ScoreModifierValidationError(modifier=values, message=f"At least one ScoreModifierOperator is required in {field.name}")
        return v

    def to_painless_script(self) -> str:
        """Convert this ScoreModifier to a painless script to modify the score."""
        mult = [x.to_painless_script("multiply_score_by") for x in self.multiply_score_by] if self.multiply_score_by is not None else []
        add = [x.to_painless_script("add_to_score") for x in self.add_to_score] if self.add_to_score is not None else []

        script = "\n".join(
            ["double additive = 0;"] + \
            mult + add + ["return Math.max(0.0, (_score + additive));"]
        )
        return f"""{script}"""