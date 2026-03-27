# General Guidelines
- All imports should be at the top of the file whenever possible.

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