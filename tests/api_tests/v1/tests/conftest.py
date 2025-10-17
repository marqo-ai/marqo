import pytest
import os


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda_test: mark test as cuda_test to skip")
    config.addinivalue_line("markers", "cpu_only_test: mark test as cpu_only_test to skip")


def pytest_collection_modifyitems(items):
    # TODO Remove this
    if not os.environ.get("TESTING_CONFIGURATION"):
        os.environ["TESTING_CONFIGURATION"] = "CUSTOM"

    if os.environ["TESTING_CONFIGURATION"] in ["CUDA_DOCKER_MARQO"]:
        # Skip cpu_only_tests if the env is CUDA
        skip_cpu_only_test = pytest.mark.skip(reason="need to not set "
                                                     "'TESTING_CONFIGURATION=CUDA_DIND_MARQO_OS' to run")
        for item in items:
            if "cpu_only_test" in item.keywords:
                item.add_marker(skip_cpu_only_test)
    else:
        # Skip cuda-tests if the env is NOT CUDA
        skip_cuda_test = pytest.mark.skip(reason="need to setEnvVars "
                                                 "'TESTING_CONFIGURATION=CUDA_DOCKER_MARQO' to run")
        for item in items:
            if "cuda_test" in item.keywords:
                item.add_marker(skip_cuda_test)