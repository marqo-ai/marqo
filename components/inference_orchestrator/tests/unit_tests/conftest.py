"""
Pytest configuration for unit tests.
"""

import os

# Set required environment variables before any imports
os.environ.setdefault("MARQO_TRITON_URL", "http://localhost:8001")
os.environ.setdefault("MARQO_MODEL_MANAGEMENT_CONTAINER_URL", "http://localhost:8002")
os.environ.setdefault("MARQO_LOG_LEVEL", "INFO")
os.environ.setdefault("MARQO_LOG_FORMAT", "plain")
