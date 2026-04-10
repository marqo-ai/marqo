# marqo-common

Lightweight shared module providing a single source of truth for model definitions across Marqo components.

## Purpose

This package centralizes the model registry so that `marqo` and `inference_orchestrator` share the same model metadata (dimensions, Triton configs, etc.) without duplication.

## Docker Build

Components that depend on `marqo-common` must include it in their Docker build context. Build from the `components/` directory:

```bash
docker build -f marqo/Dockerfile -t marqo .
docker build -f inference_orchestrator/Dockerfile -t inference-orchestrator .
```

The Dockerfiles copy `common/` into the image and install it as a local dependency.

## Adding Models

Add entries to `_MODEL_REGISTRY` in `src/marqo_common/model_registry.py`. Use `_MARQO_DEFAULT_MODELS_S3_BUCKET_PLACE_HOLDER` for S3 paths - it gets replaced at runtime.

Requires Python 3.11+. No external dependencies.
