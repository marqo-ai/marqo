import contextlib
import multiprocessing
import os

import uvicorn

from marqo import config
from marqo.api.configs import default_env_vars
from marqo.tensor_search import utils
from marqo.tensor_search.on_start_script import on_start, StartMode
from marqo.vespa.vespa_client import VespaClient
from marqo.vespa.zookeeper_client import ZookeeperClient
from marqo.tensor_search.enums import EnvVars


def generate_config() -> config.Config:
    vespa_client = VespaClient(
        config_url=utils.read_env_vars_and_defaults(EnvVars.VESPA_CONFIG_URL),
        query_url=utils.read_env_vars_and_defaults(EnvVars.VESPA_QUERY_URL),
        document_url=utils.read_env_vars_and_defaults(EnvVars.VESPA_DOCUMENT_URL),
        pool_size=utils.read_env_vars_and_defaults_ints(EnvVars.VESPA_POOL_SIZE),
        content_cluster_name=utils.read_env_vars_and_defaults(EnvVars.VESPA_CONTENT_CLUSTER_NAME),
        default_search_timeout_ms=utils.read_env_vars_and_defaults_ints(EnvVars.VESPA_SEARCH_TIMEOUT_MS),
        feed_pool_size=utils.read_env_vars_and_defaults_ints(EnvVars.VESPA_FEED_POOL_SIZE),
        get_pool_size=utils.read_env_vars_and_defaults_ints(EnvVars.VESPA_GET_POOL_SIZE),
        delete_pool_size=utils.read_env_vars_and_defaults_ints(EnvVars.VESPA_DELETE_POOL_SIZE),
        partial_update_pool_size=utils.read_env_vars_and_defaults_ints(EnvVars.VESPA_PARTIAL_UPDATE_POOL_SIZE),
    )

    # Zookeeper is only instantiated if the hosts are provided
    zookeeper_client = ZookeeperClient(
        zookeeper_connection_timeout=utils.read_env_vars_and_defaults_ints(EnvVars.ZOOKEEPER_CONNECTION_TIMEOUT),
        hosts=utils.read_env_vars_and_defaults(EnvVars.ZOOKEEPER_HOSTS)
    ) if utils.read_env_vars_and_defaults(EnvVars.ZOOKEEPER_HOSTS) else None

    # Determine default device
    default_device = utils.read_env_vars_and_defaults(EnvVars.MARQO_BEST_AVAILABLE_DEVICE)

    return config.Config(vespa_client, zookeeper_client, default_device)


_config = generate_config()


def get_config():
    return _config


def run_api_serer():
    api_worker_count = utils.read_env_vars_and_defaults_ints(EnvVars.MARQO_API_WORKER_COUNT)
    if api_worker_count < 1:
        api_worker_count = os.cpu_count() - 1

    # bind to 0.0.0.0 to expose this port in container
    uvicorn.run("api:app", host="0.0.0.0", port=8882, workers=api_worker_count)


def run_inf_app(worker_count: int):
    # bind to localhost only so it is not visible outside the container
    # TODO We will need to make sure the inference server has bootstrapped (warmed up) before start serving request
    uvicorn.run("inf_api:inf_app", host="localhost", port=8881, workers=worker_count)


@contextlib.contextmanager
def maybe_run_inference_server():
    will_run_remote_inference = utils.read_env_vars_and_defaults(EnvVars.MARQO_REMOTE_INFERENCE) == 'TRUE'
    remote_inference_url = utils.read_env_vars_and_defaults(EnvVars.MARQO_REMOTE_INFERENCE_URL)
    remote_inference_worker_count = utils.read_env_vars_and_defaults_ints(EnvVars.MARQO_INFERENCE_WORKER_COUNT)

    if will_run_remote_inference and remote_inference_url == default_env_vars()[EnvVars.MARQO_REMOTE_INFERENCE_URL]:
        # To use CUDA with multiprocessing, we must use the 'spawn' start method
        multiprocessing.set_start_method('spawn')
        p = multiprocessing.Process(target=run_inf_app, args=[remote_inference_worker_count], name='marqo-inference')
        p.start()

        try:
            yield
        finally:
            p.join()

    else:
        yield


if __name__ == "__main__":
    on_start(_config, StartMode.BOOTSTRAP)

    with maybe_run_inference_server():
        run_api_serer()
