# Marqo Inference Orchestrator

A FastAPI-based service that handles ML model inference for the Marqo tensor search engine. This service provides model loading, management, media preprocessing, and inference caching capabilities with integration to NVIDIA Triton inference server.

## Features

- Model loading and management (HuggingFace, OpenCLIP, TwelveLabs Marengo)
- Media download and preprocessing (images, text, multimodal)
- Inference caching for improved performance (LRU/LFU)
- Triton inference server integration
- OpenTelemetry instrumentation for observability
- MessagePack serialization for efficient communication

### TwelveLabs Marengo (multimodal, API-served)

[TwelveLabs](https://twelvelabs.io) Marengo is an opt-in, API-served multimodal
embedding model. Text, image and video all map into the same 512-dimensional
space, making it a cross-modal alternative to CLIP (e.g. text-to-video search).
Unlike the OpenCLIP/HuggingFace models, Marengo is not served by Triton, so it
needs no ONNX artifacts. Select it with the registered model name
`Marqo/marengo-3.0` (model `type: "twelvelabs"`), and set the
`TWELVELABS_API_KEY` environment variable. A free key with a generous free tier
is available at https://twelvelabs.io .

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management
- NVIDIA Triton inference server (for production deployments)

## Installation

Install dependencies using uv:

```bash
uv sync
```

For development with testing tools:

```bash
uv sync --group dev
```

## Usage

### Running the Service

Development mode:

```bash
PYTHONPATH=./src python -m inference_orchestrator.main
```

Production mode with uvicorn:

```bash
PYTHONPATH=./src uvicorn inference_orchestrator.main:app --host 0.0.0.0 --port 8884
```

### Docker

Build the container:

```bash
docker build -t marqo-inference .
```

Run the container:

```bash
docker run -p 8884:8884 marqo-inference
```

## API Endpoints

- `GET /` - Health check and basic information
- `POST /vectorise` - Generate embeddings from content (accepts MessagePack)
- `GET /healthz` - Liveness check for container orchestration
- `GET /models` - List loaded models
- `DELETE /models?model_name={name}` - Eject a model from memory

## Project Structure

```
inference_orchestrator/
├── src/inference_orchestrator/
│   ├── api/                              # API middleware and telemetry
│   │   ├── otel.py                       # OpenTelemetry bootstrap
│   │   └── telemetry.py                  # Telemetry middleware
│   ├── core/                             # Core configuration and logging
│   │   ├── enum.py                       # Enumerations
│   │   ├── logging.py                    # Logging configuration
│   │   └── settings.py                   # Pydantic settings
│   ├── errors/                           # Error definitions
│   │   ├── base_error.py                 # Base error class
│   │   └── common_errors.py              # Common error types
│   ├── schemas/                          # Pydantic models
│   │   ├── api.py                        # API request/response models
│   │   ├── base_model.py                 # Base Pydantic model
│   │   └── triton_channel_args.py        # Triton client configuration
│   ├── services/                         # Service layer
│   │   ├── inference_cache/              # Inference result caching
│   │   │   ├── abstract_cache.py         # Cache interface
│   │   │   ├── caching_inference.py      # Cached inference wrapper
│   │   │   ├── marqo_inference_cache.py  # Main cache implementation
│   │   │   ├── marqo_lfu_cache.py        # LFU cache strategy
│   │   │   ├── marqo_lru_cache.py        # LRU cache strategy
│   │   │   └── monitoring.py             # Cache metrics
│   │   ├── media_download_and_preprocess/  # Media processing
│   │   │   ├── image_download.py         # Image downloading
│   │   │   ├── media_download_and_preprocess.py  # Main processing
│   │   │   └── split_text.py             # Text chunking
│   │   ├── triton_inference/             # Triton integration
│   │   │   ├── embedding_models/         # Model-specific implementations
│   │   │   │   ├── hugging_face/         # HuggingFace models
│   │   │   │   ├── open_clip/            # OpenCLIP models
│   │   │   │   ├── random/               # Random models (testing)
│   │   │   │   ├── twelvelabs/            # TwelveLabs Marengo (API-served)
│   │   │   │   ├── abstract_embedding_model.py
│   │   │   │   ├── abstract_preprocessor.py
│   │   │   │   ├── base_model_properties.py
│   │   │   │   ├── data_type_conversion.py
│   │   │   │   ├── marqo_model_registry.py
│   │   │   │   ├── model_download_cache.py
│   │   │   │   ├── model_properties_parser.py
│   │   │   │   └── url_parser.py
│   │   │   ├── inference_pipelines/      # Inference orchestration
│   │   │   │   ├── abstract_inference_pipeline.py
│   │   │   │   ├── hugging_face_model_inference_pipeline.py
│   │   │   │   ├── open_clip_model_inference_pipeline.py
│   │   │   │   ├── random_model_inference_pipeline.py
│   │   │   │   └── twelvelabs_model_inference_pipeline.py
│   │   │   ├── model_manager/            # Model lifecycle management
│   │   │   │   ├── model_management_client.py
│   │   │   │   └── model_manager.py
│   │   │   ├── triton/                   # Triton client wrappers
│   │   │   │   ├── input_type.py
│   │   │   │   └── triton_grpc_client.py
│   │   │   └── triton_inference.py       # Main inference service
│   │   └── errors.py                     # Service-level errors
│   ├── config.py                         # Configuration management
│   ├── main.py                           # FastAPI application entry point
│   ├── marqo_docs.py                     # Documentation utilities
│   ├── on_start_script.py                # Startup initialization
│   └── version.py                        # Version information
├── tests/
│   ├── integration_tests/                # Integration tests
│   └── unit_tests/                       # Unit tests
├── pyproject.toml                        # Project dependencies and config
└── README.md                             # This file
```

## Configuration

Configuration is managed via environment variables and Pydantic settings. Key environment variables:

- `TRITON_SERVER_URL` - URL of the Triton inference server
- `CACHE_SIZE` - Maximum number of cached inference results
- `CACHE_STRATEGY` - Cache eviction strategy (LRU/LFU)
- `LOG_LEVEL` - Logging level (DEBUG/INFO/WARNING/ERROR)
- `OTEL_ENABLED` - Enable OpenTelemetry instrumentation

See `src/inference_orchestrator/core/settings.py` for all available settings.

## Testing

Run unit tests:

```bash
PYTHONPATH=./src pytest tests/unit_tests/ -v
```

Run integration tests:

```bash
PYTHONPATH=./src pytest tests/integration_tests/ -v
```

Run specific test file:

```bash
PYTHONPATH=./src pytest tests/unit_tests/services/inference_cache/test_cache.py -v
```

## Architecture

### Core Components

1. **API Layer** (`api/`)
   - FastAPI application with OpenTelemetry instrumentation
   - MessagePack serialization for efficient data transfer
   - Custom exception handlers and middleware

2. **Service Layer** (`services/`)
   - **Inference Cache**: LRU/LFU caching with monitoring and eviction strategies
   - **Media Processing**: Downloads and preprocesses images/text for model input
   - **Triton Inference**: Manages model lifecycle and inference requests to Triton server

3. **Model Management** (`services/triton_inference/`)
   - **Embedding Models**: Abstract interfaces and implementations for various model types
   - **Inference Pipelines**: Orchestrates preprocessing, inference, and postprocessing
   - **Model Manager**: Handles model loading, unloading, and lifecycle management

4. **Configuration** (`core/`)
   - Pydantic-based settings with environment variable support
   - Structured logging with JSON output
   - Singleton pattern via `@lru_cache()` decorators

### Request Flow

1. Client sends inference request (MessagePack encoded) to `/vectorise`
2. Request validated via Pydantic schemas
3. Cache check for existing results
4. If cache miss:
   - Media downloaded and preprocessed
   - Appropriate inference pipeline selected
   - Model loaded (if needed) via model manager
   - Inference executed on Triton server
   - Results cached
5. Response returned (MessagePack encoded)


## Error Handling

All errors inherit from `BaseMarqoInferenceError` in `errors/base_error.py`. Service-level errors are defined in `services/errors.py`:

- `ServiceError` - Base class for service errors
- `InternalServerError` - Internal server errors (500)
- Model-specific errors in respective model implementation files