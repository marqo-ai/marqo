import multiprocessing

import uvicorn

from marqo import config
from marqo.tensor_search import utils
from marqo.tensor_search.on_start_script import on_start
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


def run_inf_app():
    uvicorn.run("inf_api:inf_app", host="localhost", port=8881)


def run_api():
    uvicorn.run("api:app", host="localhost", port=8882, workers=2)


if __name__ == "__main__":
    on_start(_config, 'main')
    p = multiprocessing.Process(target=run_inf_app, name='marqo-inference')
    p.start()

    run_api()

    p.join()