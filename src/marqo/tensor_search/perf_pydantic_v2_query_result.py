import time

import orjson

from marqo.core.models.marqo_index import SemiStructuredMarqoIndex, Model, TextPreProcessing, \
    TextSplitMethod, ImagePreProcessing, VideoPreProcessing, AudioPreProcessing, DistanceMetric, VectorNumericType, \
    HnswConfig, Field, FieldType, FieldFeature, StringArrayField, TensorField
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
from marqo.tensor_search.tensor_search import gather_documents_from_response
from marqo.version import get_version
from marqo.vespa.models import QueryResult

if __name__ == '__main__':
    result_file = '/Users/Yihan/Downloads/new_limit_60.json'
    with open(result_file, 'r') as f:
        result_dict = orjson.loads(f.read())

    marqo_index = SemiStructuredMarqoIndex(
        name='index',
        schema_name='index_schema',
        model=Model(name='hf/all_datasets_v4_MiniLM-L6'),
        normalize_embeddings=True,
        text_preprocessing=TextPreProcessing(
            split_length=2,
            split_overlap=0,
            split_method=TextSplitMethod.Sentence
        ),
        image_preprocessing=ImagePreProcessing(
            patch_method=None
        ),
        video_preprocessing=VideoPreProcessing(
            split_length=20,
            split_overlap=1,
        ),
        audio_preprocessing=AudioPreProcessing(
            split_length=20,
            split_overlap=1,
        ),
        distance_metric=DistanceMetric.Angular,
        vector_numeric_type=VectorNumericType.Float,
        hnsw_config=HnswConfig(
            ef_construction=128,
            m=16
        ),
        marqo_version=get_version(),
        created_at=time.time(),
        updated_at=time.time(),
        treat_urls_and_pointers_as_images=True,
        filter_string_max_length=50,
        lexical_fields=[
            Field(name=field_name, type=FieldType.Text,
                  features=[FieldFeature.LexicalSearch],
                  lexical_field_name=f'{SemiStructuredVespaSchema.FIELD_INDEX_PREFIX}{field_name}'
                  ) for field_name in [
                "color",
                "meta",
                "product_type",
                "all_inventory",
                "product_image_plus",
                "image_url",
                "title",
                "body_html_safe",
                "handle",
                "productname",
                "product_image",
                "inventory_policy",
                "image",
                "vendor",
                "option_name",
                "compare_at_price",
                "title_es",
                "properties_concatenated",
                "product_image_trending",
                "product_image_plus_trending",
                "product_video",
                "message",
            ]
        ],  # : List[Field]
        tensor_fields=[
            TensorField(
                name=field_name,
                chunk_field_name=f'{SemiStructuredVespaSchema.FIELD_CHUNKS_PREFIX}{field_name}',
                embeddings_field_name=f'{SemiStructuredVespaSchema.FIELD_EMBEDDING_PREFIX}{field_name}',
            ) for field_name in [
                "image_title_multimodal"
            ]
        ],  # : List[TensorField]
        string_array_fields=[
            StringArrayField(
                name=field_name, type=FieldType.ArrayText, features=[FieldFeature.Filter],
                string_array_field_name=f'{SemiStructuredVespaSchema.FIELD_STRING_ARRAY_PREFIX}{field_name}'
            ) for field_name in [
                "named_tags_names",
                "collections",
                "available_markets",
                "all_sizes_in_stock_array",
                "tags",
                "named_tags",
                "all_sizes_array",
            ]
        ],  # : Optional[List[StringArrayField]]
    )

    iterations = 10_000
    start = time.perf_counter()

    for _ in range(iterations):
        query_result = QueryResult(**result_dict)
        gathered_docs = gather_documents_from_response(query_result, marqo_index, highlights=False,
                                                       attributes_to_retrieve=None)

    end = time.perf_counter()
    total_time = end - start
    avg_time = total_time / iterations

    print(f"Total time for {iterations} iterations: {total_time:.6f} seconds")
    print(f"Average time per iteration: {avg_time:.9f} seconds")

