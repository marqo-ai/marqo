import pytest



def pytest_addoption(parser):
    parser.addoption("--lm", "--largemodel", action="store_true", default=False, type = bool)