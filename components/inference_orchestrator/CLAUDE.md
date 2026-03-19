# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Marqo Inference Container is a FastAPI-based service that handles ML model inference for the Marqo tensor search engine. It provides:

- Model loading and management (HuggingFace, OpenCLIP, etc.)
- Media download and preprocessing (images, text, multimodal)
- Inference caching for improved performance
- Triton inference server integration
- OpenTelemetry instrumentation

## Development Commands

### Environment Setup
- Use `uv` for dependency management
- Python version: 3.11+
- Set `PYTHONPATH=./src` for running tests and scripts
- Install dependencies: `uv sync`
- Install dev dependencies (if applicable): `uv sync --group dev`

# Project Structure

```
src/marqo_inference_container/
├── api/                    # API middleware and OpenTelemetry setup
├── core/                   # Core settings and logging
├── errors/                 # Error definitions and base error classes
├── schemas/                # Pydantic models for API requests/responses
├── services/
│   ├── inference_cache/    # Caching layer for inference results
│   ├── media_download_and_preprocess/  # Media processing pipeline
│   ├── model_download/     # HuggingFace model downloading
│   └── triton_inference/   # Triton server integration
├── config.py              # Configuration management
├── main.py                # FastAPI application entry point
├── on_start_script.py     # Startup initialization tasks
└── version.py             # Version information
```

### Testing
- Run unit tests: `PYTHONPATH=./src pytest tests/unit_tests/ -v`
- Tests are located in `tests/unit_tests/` and follow the same package hierarchy as source code
- If you add new tests or change any tests, make sure to run them and verify they pass
- Prefer updating existing tests over creating new ones
- Use subtests for grouping related tests with shared setup

### Running the Application
- Development: `PYTHONPATH=./src python -m marqo_inference_container.main`
- Docker: Build with `docker build -t marqo-inference .`
- Environment variables can be set via `.env` file or exported in shell

## Architecture

### Core Components

1. **API Layer** (`src/marqo_inference_container/`)
   - `main.py`: FastAPI application entry point
   - `api/`: Middleware and OpenTelemetry setup
   - `schemas/`: Pydantic models for request/response validation

2. **Service Layer** (`src/marqo_inference_container/services/`)
   - `inference_cache/`: LRU caching of inference results with monitoring
   - `media_download_and_preprocess/`: Downloads and preprocesses images/text for model input
   - `triton_inference/`: Integration with NVIDIA Triton inference server
   - `model_download/`: Downloads models from HuggingFace hub

3. **Configuration** (`src/marqo_inference_container/core/`)
   - `settings.py`: Pydantic settings with environment variable support
   - `config.py`: Configuration management
   - Uses `@lru_cache()` for singleton pattern where applicable

### API Endpoints

Main endpoint is `/infer` which accepts `InferenceRequest` (JSON or msgpack) and returns embeddings.

### Error Handling

- All errors inherit from `BaseMarqoInferenceError` in `errors/base_error.py`
- Specific error types in `errors/inference_errors.py` and `errors/common_errors.py`

### Configuration

Configuration is managed via Pydantic settings in `core/settings.py` and loaded through `config.py`.

## Development Guidelines

- Follow existing package structure for new modules
- Unit tests must mirror the source package hierarchy
- Use dependency injection pattern via FastAPI's `Depends()` where applicable
- All configuration should be environment-variable driven
- Use structured logging with appropriate log levels
- Always import everything at the top of the file, avoid inline imports unless necessary to prevent circular dependencies

### Detailed Test Development Guidelines
- Each source package should have a corresponding test package
- Use subtests for related test cases with shared setup
- Mock external dependencies (e.g., Triton server, model downloads) for unit tests
- Ensure high test coverage for critical components
- Review the tests to ensure there is no duplicated test logic
- When using `assertEqual`, put the expected value first, and the actual value second
- When using subtests, group all the test cases into a list of tuples first with message, input, expected output, then loop through the list and call `self.subTest` for each case
- Add docstrings if possible to explain the purpose of the test case
- When doing assert on the expected values, be more specific, e.g. check the length of a list, check if a string contains a substring, check if the value is of a certain type, etc. Avoid generic `assertTrue`, `assertFalse`, `assertNone`, `assertNotNone` unless absolutely necessary
- When you need to test things regarding environment variables, take care of the `.env` file in the root folder as it may affect the test results. So make sure to set the environment variables explicitly in the test case if needed
- Avoid testing non-public methods unless absolutely necessary. If you need to test a private method, consider if it should be made public or if the functionality can be tested through a public method
- In unit tests, avoid using `time.sleep` or any other blocking calls that may slow down the test execution. Instead, use mocking to simulate delays or timeouts
- Avoid running integration tests unless explicitly requested, as they may need to download a lot of models and take a long time to run


## Important Instruction Reminders

- Do what has been asked; nothing more, nothing less
- NEVER create files unless absolutely necessary
- ALWAYS prefer editing an existing file to creating a new one
- NEVER proactively create documentation files unless explicitly requested
