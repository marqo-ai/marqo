# Marqo Development Guide for Claude Code

## Overview

Marqo is an end-to-end vector search engine for text and images that bundles ML model inference with vector storage and
retrieval. It provides a "documents in, documents out" approach, handling embedding generation, preprocessing, and
search through a single API.

## Quick Start Commands

### Development Environment Setup

```bash
# Create and activate virtual environment
python -m venv ./venv
source ./venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For development dependencies
pip install -r requirements.dev.txt
```

### Starting Marqo Locally

```bash
# Option 1: Docker (recommended for testing)
docker rm -f marqo
docker pull marqoai/marqo:latest
docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest

# Option 2: Local development (requires Vespa setup)
export PYTHONPATH=./src
export MARQO_ENABLE_BATCH_APIS=true
export MARQO_MODE=COMBINED
python src/marqo/tensor_search/api.py
```

### Vespa Backend Setup

Marqo requires Vespa as its vector database backend:

```bash
# Start local Vespa instance
python scripts/vespa_local/vespa_local.py full_start

# Check Vespa status
docker ps | grep vespa
curl -f http://localhost:8080/ApplicationStatus
```

## Testing Structure & Commands

### Environment Variables for Tests

```bash
# Unit and Integration tests
export PYTHONPATH=./src

# API tests
export PYTHONPATH=./tests/api_tests/v1/tests/api_tests
export MARQO_ENABLE_BATCH_APIS=true
export MARQO_MODE=COMBINED
```

### Test Execution

```bash
# Unit Tests (fast, isolated)
export PYTHONPATH=./src
pytest tests/unit_tests/

# Integration Tests (requires Vespa)
export PYTHONPATH=./src
pytest tests/integ_tests/

# API Tests (requires running Marqo API)
# Terminal 1: Start API
export PYTHONPATH=./src
python src/marqo/tensor_search/api.py

# Terminal 2: Run tests
export PYTHONPATH=./tests/api_tests/v1/tests/api_tests
pytest tests/api_tests/v1/tests/api_tests/

# Performance Tests
cd perf_tests
pip install -r requirements.txt
locust  # Uses locust.conf settings
```

### Test Dependencies

- **Unit tests**: No external dependencies
- **Integration tests**: Requires Vespa running locally
- **API tests**: Requires both Vespa and Marqo API running
- Use `MarqoTestCase` from `tests.integ_tests.marqo_test` for integration tests

## Architecture Overview

### Core Components

- **Tensor Search Engine**: `src/marqo/tensor_search/` - Main search implementation
- **Inference Engine**: `src/marqo/core/inference/` - ML model inference and modality detection
- **Vespa Integration**: `src/marqo/vespa/` - Vector database client
- **API Layer**: `src/marqo/tensor_search/api.py` - FastAPI HTTP endpoints

### Index Types

1. **Unstructured**: Flexible schema, automatic field detection. This is a legacy index type kept for backwards
   compatibility. Most of the time, when we talk about unstructured indexes, we are referring to semi-structured indexes
   which supersede unstructured indexes. Users can't create new indexes of this type.
2. **Structured**: Predefined schema with strict field types
3. **Semi-structured**: Hybrid approach with optional schema definitions

### Search Methods

- **TENSOR**: Semantic/vector search using ML embeddings
- **LEXICAL**: Traditional keyword-based search
- **HYBRID**: Combination with ranking fusion (RRF - Reciprocal Rank Fusion)

### Modality Support

- **Text**: Natural language processing via various embedding models
- **Images**: Vision models supporting URLs, file paths, and base64 encoding
- **Multimodal**: Combined text and image queries and indexing

## Key Development Areas

### Image Processing Pipeline

- **Modality Detection**: `src/marqo/core/inference/modality_utils.py` - Determines content type
- **Image Loading**: `src/marqo/inference/media_download_and_preprocess/image_download.py`
- **Base64 Support**: Recently implemented for both data URLs and plain base64 strings
- **Priority Order**: Base64 → URL → File path detection

### Inference Systems

- **Native Inference**: `src/marqo/inference/native_inference/` - Local model execution
- **S2 Inference**: `src/marqo/s2_inference/` - Legacy inference system
- **Caching**: `src/marqo/inference/inference_cache/` - LRU/LFU caching for embeddings

### Vespa Index Management

Each index type has dedicated handlers:

- `src/marqo/core/unstructured_vespa_index/`
- `src/marqo/core/structured_vespa_index/`
- `src/marqo/core/semi_structured_vespa_index/`

## Environment Configuration

### Key Environment Variables

```bash
# Marqo Operation Mode
MARQO_MODE=COMBINED  # COMBINED, API, or INFERENCE

# Vespa Configuration
VESPA_QUERY_URL=http://localhost:8080
VESPA_DOCUMENT_URL=http://localhost:8080
VESPA_CONFIG_URL=http://localhost:19071
ZOOKEEPER_HOSTS=localhost:2181

# Feature Flags
MARQO_ENABLE_BATCH_APIS=true
MARQO_ENABLE_THROTTLING=FALSE

# Inference Settings
MARQO_API_INFERENCE_CACHE_SIZE=100
MARQO_API_INFERENCE_CACHE_TYPE=lru

# Logging
MARQO_LOG_LEVEL=info
```

## Development Workflow

### Branch Structure

- **Main branch**: `mainline`
- **Feature branches**: Typically `username/feature-description`

### Pre-commit Requirements

1. All tests must pass: `pytest tests/unit_tests/ tests/integ_tests/`
2. Code follows existing patterns and conventions
3. New features require corresponding tests

### Creating Pull Requests

```bash
# Ensure tests pass
pytest tests/unit_tests/
pytest tests/integ_tests/ 

# Create PR against mainline branch
# Delete feature branch after merge
```

## Common Development Patterns

### Error Handling

- **Internal errors**: Raise `InternalError` or subclasses
- **S2 Inference errors**: Raise `S2InferenceError`
- **User-facing errors**: Use appropriate API exceptions

### Code Style

- Explicitly state argument names: `func(a=1, b=2)` vs `func(1, 2)`
- Follow existing module structure and naming conventions
- Comprehensive test coverage at unit, integration, and API levels

### Adding New Features

1. Create unit tests first (TDD approach)
2. Implement core functionality
3. Add integration tests
4. Add API tests if exposing new endpoints
5. Update documentation

## Performance Considerations

### Model Loading

- Models are cached and reused across requests
- Use `MARQO_MODELS_TO_PRELOAD` to warm frequently used models
- Monitor memory usage with built-in profiling tools

### Vespa Optimization

- Index sharding for large datasets
- Proper schema design for search performance
- Connection pooling configured via environment variables

### Search Performance

- HNSW vector indexing for fast similarity search
- Hybrid search combines speed of lexical with accuracy of semantic search
- Score modifiers and filtering can impact performance

## Troubleshooting

### Common Issues

- **Import errors**: Check `PYTHONPATH` is set correctly
- **Vespa connection failures**: Ensure Vespa is running and accessible
- **Model loading errors**: Check model cache and download permissions
- **API startup failures**: Verify all dependencies are installed and ports are available

### Debug Tools

- **Logging**: Adjust `MARQO_LOG_LEVEL` for detailed output
- **Memory profiling**: Built-in tools in `src/marqo/core/monitoring/`
- **Health endpoints**: `/health` for API status checks

This guide focuses on practical development information. For user-facing documentation, API references, and deployment
guides, see the main README.md and official documentation at https://docs.marqo.ai/.