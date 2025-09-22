# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Marqo Model Management Container is a FastAPI-based service that manages machine learning models for Triton Inference Server. It handles model downloading, loading, and unloading operations.

## Development Commands

### Environment Setup
- Use `uv` for dependency management
- Set `PYTHONPATH=./src` for running tests and scripts
- Install dependencies: `uv sync`
- Install dev dependencies: `uv sync --group dev`

### Testing
- Run unit tests: `PYTHONPATH=./src uv run pytest tests/unit_tests/ -v`
- Tests are located in `tests/unit_tests/` and follow the same package hierarchy as source code
- Use subtests for grouping related tests with shared setup

### Running the Application
- Development: `PYTHONPATH=./src python -m marqo_model_management_container.main`
- Docker: Build with `docker build -t marqo-model-management .`
- Application runs on port 8883 by default

## Architecture

### Core Components

1. **API Layer** (`src/marqo_model_management_container/api/`)
   - `main.py`: FastAPI application entry point
   - `v1_routes.py`: API endpoints for model operations
   - `lifespan.py`: Application startup/shutdown handlers
   - `exception_handlers.py`: Global error handling
   - `request_id.py`: Request tracking middleware

2. **Service Layer** (`src/marqo_model_management_container/service/`)
   - `model_manager/`: Handles model downloading and Triton config generation
   - `triton/`: Client for communicating with Triton Inference Server

3. **Configuration** (`src/marqo_model_management_container/core/`)
   - `settings.py`: Pydantic settings with environment variable support
   - `config.py`: Dependency injection configuration
   - Uses `@lru_cache()` for singleton pattern

### Key Services

- **ModelManager**: Downloads models and generates Triton config.pbtxt files using Jinja2 templates
- **TritonClient**: HTTP client for Triton Inference Server API (`/v2/repository/models/`)
- **TritonModelDownloader**: Handles model file downloads from various sources

### Configuration

Environment variables (see `env.example`):
- `TRITON_URL`: Triton server endpoint (required)
- `MODEL_BASE_DIR`: Local model storage path (default: `./cache/models`)
- `MARQO_MODELS_TO_PRELOAD`: JSON array of models to load on startup
- `LOG_LEVEL`: Logging level (debug|info|warning|error)
- `LOG_FORMAT`: Log format (text|json)

### API Endpoints

- `POST /v1/models/load`: Load a model into Triton
- `POST /v1/models/{model_name}/unload`: Unload a model from Triton

### Schema Design

- Uses Pydantic models for request/response validation
- `TritonModelProperties`: Defines model configuration for Triton
- `LoadModelRequest`: API request wrapper
- Custom error handling with structured problem responses

### Dependencies

- **FastAPI**: Web framework
- **Pydantic**: Data validation and settings
- **httpx**: HTTP client for Triton communication
- **Jinja2**: Template engine for Triton config generation
- **fsspec/s3fs**: File system abstraction for model downloads

## Development Guidelines

- Follow existing package structure for new modules
- Unit tests must mirror the source package hierarchy
- Use dependency injection pattern via FastAPI's `Depends()`
- All configuration should be environment-variable driven
- Use structured logging with request IDs
- Models are downloaded to `MODEL_BASE_DIR` and organized by model name

### Detailed Test Development Guidelines
- Each source package should have a corresponding test package
- Use subtests for related test cases with shared setup
- Mock external dependencies (e.g., Triton server) for unit tests
- Ensure high test coverage for critical components
- Review the tests to ensure there is no duplicated test logic
- When using assertEqual, put the expected value first, and the actual value second
- When using subtests, group all the test cases into a list of tuples first with message, input, expected output, then loop through the list and call self.subTest for each case
- Add doc string if possible to explain the purpose of the test case