import pytest

def pytest_addoption(parser):
    parser.addoption("--largemodel", action="store_true", default = False)