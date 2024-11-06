import json
import os
import sys
from pathlib import Path

from marqo_test import MarqoTestCase

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class BaseCompatibilityTestCase(MarqoTestCase):
    """
    Base class for backwards compatibility tests. Contains a prepare method that should be implemented by subclasses to
    add documents / prepare marqo state. Also contains methods to save and load results to/from a file so that
    test results can be compared across versions.
    """
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    @classmethod
    def get_results_file_path(cls):
        """Dynamically generate a unique file path based on the class name."""
        return Path(f"{cls.__qualname__}_stored_results.json")

    @classmethod
    def save_results_to_file(cls, results):
        """Save results to a JSON file."""
        filepath = cls.get_results_file_path()
        with filepath.open('w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {filepath}")

    @classmethod
    def load_results_from_file(cls):
        """Load results from a JSON file."""
        filepath = cls.get_results_file_path()
        with filepath.open('r') as f:
            results = json.load(f)
        print(f"Results loaded from {filepath}")
        return results

    def prepare(self):
        """Prepare marqo state like adding documents"""
        pass