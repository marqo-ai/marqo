from typing import List

from marqo.core.models.marqo_index import MarqoIndex, FieldType, FieldFeature

_marqo_vepsa_type_mapping = {
    FieldType.Text: "string",
    FieldType.Bool: "bool",
    FieldType.Int: "int",
    FieldType.Float: "float",
    FieldType.ArrayText: "array<string>",
    FieldType.ArrayBool: "array<bool>",
    FieldType.ArrayInt: "array<int>",
    FieldType.ArrayFloat: "array<float>",
    FieldType.ImagePointer: "string"
}


def generate_schema(marqo_index: MarqoIndex) -> str:
    pass


def _generate_fields(marqo_index: MarqoIndex, schema: List[str]) -> None:
    for field in marqo_index.fields:
        schema.append(f"field {field.name} type {_get_vespa_type(field.type)} {{")
        # indexing
        indexing = ['summary']
        if FieldFeature.LexicalSearch in field.features:
            indexing.append('index')
        if FieldFeature.Filter in field.features or FieldFeature.ScoreModifier in field.features:
            indexing.append('attribute')

        schema.append(f'indexing: {" | ".join(indexing)}')

        # index
        if FieldFeature.LexicalSearch in field.features:
            schema.append('index: enable-bm25')

        # attribute
        if FieldFeature.Filter in field.features:
            schema.append('attribute: fast-search')

        # rank
        if FieldFeature.Filter in field.features:
            schema.append('rank: filter')

        schema.append('}')

    model_dim = _get_model_dimension(marqo_index.model)
    for field in marqo_index.tensor_fields:
        schema.append(f'field __marqo_chunks_{field} type array<string> {{')
        schema.append(f'indexing: attribute | summary')
        schema.append()


def _get_vespa_type(marqo_type: FieldType) -> str:
    try:
        return _marqo_vepsa_type_mapping[FieldType(marqo_type)]
    except KeyError:
        raise ValueError(f"Unknown Marqo type: {marqo_type}")


def _get_model_dimension(model: str) -> int:
    pass
