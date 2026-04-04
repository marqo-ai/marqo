"""
Manually bootstrap Vespa with the current jar and configuration.

Run this from the Marqo container (or locally with PYTHONPATH=src) after
building the jar. It creates a minimal Config and triggers bootstrap_vespa(),
which copies the jar from vespa/target/ into the Vespa application package
and deploys it.

Usage:
    PYTHONPATH=src python scripts/bootstrap_vespa.py

Environment variables (uses defaults from EnvVars if not set):
    VESPA_CONFIG_URL    - Vespa config server URL (default: http://localhost:19071)
    VESPA_QUERY_URL     - Vespa query URL (default: http://localhost:8080)
    VESPA_DOCUMENT_URL  - Vespa document URL (default: http://localhost:8080)
"""

import sys
import os

# Add src to path if running from scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from marqo.tensor_search.on_start_script import BootstrapVespa
from marqo.tensor_search.api import generate_config


def main():
    print("Creating config...")
    config = generate_config()

    print("Bootstrapping Vespa (deploying jar + application package)...")
    bootstrapper = BootstrapVespa(config)
    bootstrapper.run()

    print("Done.")


if __name__ == "__main__":
    main()
