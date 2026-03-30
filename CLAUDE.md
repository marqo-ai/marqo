# Goal - Wishlist Feature
1. The customer wants to be able to input a list of product _id to filter over. Results will be restricted to these IDs.
2. We need to support list size up to 1000.
3. Nice to have would be size 7670.
2. This is similar to the "IN Filter" we have in structured indexes, but not semi-structured.
3. Easy win: Implement "IN filter" for semi-structured indexes.

Issue:
1. There is a StackOverflow issue issue with Vespa when filter is very long
Trying a pattern like this: `_id:6540013895804 OR _id:6540013895805 OR _id:6540013895806...`
We hit the StackOverflow issue with the filter at 320~ _ids. Error looks like this:
```
raise VespaStatusError(message=resp.text, cause=e) from e","marqo.vespa.exceptions.VespaStatusError: 500:
{\"root\":{\"relevance\":1.0,\"fields\":{\"totalCount\":0},\"errors\":[{\"code\":6,\"summary\":\"Error in plugin
Searcher\",\"message\":\"Error in 'execution of chain 'marqo'': StackOverflowError\",\"stackTrace\":\
"java.lang.StackOverflowError\\n\\tat org.antlr.v4.runtime.RuleContext.getText(RuleContext.java:135)\\n\\tat
java.base/java.util.stream.ReferencePipeline$3$1.accept(ReferencePipeline.java:197)\\n\\tat
```

2. In March 2024, IN filter could not be implemented for unstructured indexes because of this limitation:
While attempting to use the IN filter here on both marqo__short_string_fields.value and marqo__string_array:
{'yql': 'select * from a862ea655f2e7490e8b45e33eb2df8874 where ({targetHits:3, approximate:True, hnsw.exploreAdditionalHits:1997}nearestNeighbor(marqo__embeddings, embedding_query)) AND ((marqo__short_string_fields contains sameElement(key contains "list_field_1", value in (\'tag1\', \'tag2\'))) OR (marqo__string_array in (\'list_field_1::tag1\', \'list_field_1::tag2\')))', 'hits': 3, 'ranking': 'embedding_similarity', 'model.restrict': 'a862ea655f2e7490e8b45e33eb2df8874', 'input.query(embedding_query)':

I getting the error
400: {"root":{"id":"toplevel","relevance":1.0,"fields":{"totalCount":0},"errors":[{"code":4,"summary":"Invalid query parameter","message":"Could not create query from YQL: The in operator is only supported for integer and string fields. The field value is not of these types"leading me to believe that the "value" field may not be properly assigned as a string in the schema. Looking into this.

- Is this still an issue with our current version of Vespa?
- Since we now use semi-structured indexes, is it possible to implement this now?
- Explain why or why not.


Questions:
1. How much effort is it to implement "IN filter" for semi-structured indexes?
2. Find the Vespa limits (how many _id in the list before it breaks?)
3. How can we get around these limits to get list length 1000 and up? Use Vespa repo and documentation:
https://search.vespa.ai/ and https://github.com/vespa-engine/vespa

# General Guidelines
- All imports should be at the top of the file whenever possible.
- We are deprecating `structured_vespa_index` so make all your changes directly to `semi_structured_vespa_index`
  even if it supposedly inherits from `structured_vespa_index`.

# Environment Setup

Make sure the virtual environment is activated before running any commands.
Use the environment variables in .env and the virtual environment in .venv.
If you make changes to the searcher (HybridSearcher.java), make sure to build it with `mvn clean package` and redeploy
the application package to Vespa before trying again.

# Tests

- Unit tests are in ./tests/unit_tests, integ tests are in ./tests/integ_tests, API tests are in
  ./tests/api_tests/v1/tests/api_tests
- To run unit and integ tests, make sure working directory is repo root and set PYTHONPATH=./src.
- If running integ or API tests, make sure Vespa is running via docker ps. If not running, use
  python scripts/vespa_local/vespa_local.py full_start to run Vespa first.
- To run API tests, first run Marqo API in one process by running src/marqo/tensor_search/api.py using
  PYTHONPATH=./src MARQO_ENABLE_BATCH_APIS=true MARQO_MODE=COMBINED. While the API is running, run API tests via pytest
  using PYTHONPATH=./tests/api_tests/v1/tests/api_tests . If Marqo API fails to run, stop. Terminate Marqo API when
  done.
- Unit tests must follow the same package hierarchy as the code they test.
- If you add new tests or change any tests, make sure to run them and verify they pass.
- If there are existing tests, prefer to update them to cover the changes over creating new tests.
- Use subtests to group tests together where appropriate, especially for tests that share setup code.

# Core Components

- **Tensor Search Engine**: `src/marqo/tensor_search/` - Main search implementation
- **Inference Engine**: `src/marqo/core/inference/` - ML model inference and modality detection
- **Vespa Integration**: `src/marqo/vespa/` - Vector database client
- **API Layer**: `src/marqo/tensor_search/api.py` - FastAPI HTTP endpoints

# Index Types

1. **Unstructured**: Flexible schema, automatic field detection. This is a legacy index type kept for backwards
   compatibility. Most of the time, when we talk about unstructured indexes, we are referring to semi-structured indexes
   which supersede unstructured indexes. Users can't create new indexes of this type.
2. **Structured**: Predefined schema with strict field types
3. **Semi-structured**: Hybrid approach with optional schema definitions

# Search Methods

- **TENSOR**: Semantic/vector search using ML embeddings
- **LEXICAL**: Traditional keyword-based search
- **HYBRID**: Combination with ranking fusion (RRF - Reciprocal Rank Fusion)

# Vespa Index Management

Each index type has dedicated handlers:

- `src/marqo/core/unstructured_vespa_index/`
- `src/marqo/core/structured_vespa_index/`
- `src/marqo/core/semi_structured_vespa_index/`

# Branch Structure

- **Main branch**: `mainline`
- **Feature branches**: Typically `username/feature-description`

# Errors

- Core classes must only raise marqo.core.exceptions or marqo.exceptions, never marqo.api.exceptions. The mapping to
  API exceptions is done in the API layer.