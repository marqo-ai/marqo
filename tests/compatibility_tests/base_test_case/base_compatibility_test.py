import json
import logging
from abc import abstractmethod, ABC
from pathlib import Path

from tests.compatibility_tests.base_test_case.marqo_test import MarqoTestCase
from tests.compatibility_tests.compatibility_test_logger import get_logger


class BaseCompatibilityTestCase(MarqoTestCase, ABC):
    """
    Base class for backwards compatibility tests. Contains a prepare method that should be implemented by subclasses to
    add documents / prepare marqo state. Also contains methods to save and load results to/from a file so that
    test results can be compared across versions.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not hasattr(cls, 'logger'):
            cls.logger = get_logger(f"tests.compatibility_tests.{cls.__module__}.{cls.__name__}")

    @classmethod
    def get_results_file_path(cls):
        """Dynamically generate a unique file path based on the class name."""
        return Path(f"{cls.__qualname__}_stored_results.json")

    @classmethod
    def tearDownClass(cls) -> None:
        # A function that will be automatically called after each test call
        # This removes all the loaded models. It will also remove all the indexes inside a marqo instance.
        # Be sure to set the indexes_to_delete list with the indexes you want to delete, in the test class.
        if cls.indexes_to_delete:
            cls.logger.debug(f"Deleting indexes: {cls.indexes_to_delete}")
        super().tearDownClass()
        cls.delete_file()

    @classmethod
    def save_results_to_file(cls, results):
        """Save results to a JSON file."""
        filepath = cls.get_results_file_path()
        with filepath.open('w') as f:
            json.dump(results, f, indent=4)
        cls.logger.debug(f"Results saved to {filepath}")

    @classmethod
    def load_results_from_file(cls):
        """Load results from a JSON file."""
        filepath = cls.get_results_file_path()
        with filepath.open('r') as f:
            results = json.load(f)
        cls.logger.debug(f"Results loaded from {filepath}")
        return results

    @classmethod
    def delete_file(cls):
        """Delete the results file."""
        filepath = cls.get_results_file_path()
        if filepath.exists():
            filepath.unlink()
            cls.logger.debug(f"Results file deleted: {filepath}")
        else:
            cls.logger.debug(f"Not deleting, as the results file was never created in the first place.")

    @abstractmethod
    def prepare(self):
        """Prepare marqo state like adding documents"""
        pass

    @classmethod
    def set_logging_level(cls, level: str):
        """Set the logging level for this class's logger"""
        log_level = getattr(logging, level.upper(), None)
        if log_level is None:
            raise ValueError(f"Invalid log level: {level}. Using current log level: {logging.getLevelName(cls.logger.level)}.")
        cls.logger.setLevel(log_level)
        cls.logger.info(f"Logging level changed to {level.upper()}")