from marqo.vespa.marqo_zookeeper_client import MarqoZookeeperClient
from marqo.core.distributed_lock.zookeeper_distributed_lock import ZookeeperDistributedLock

_DEPLOYMENT_LOCK_PATH = "/marqo__deployment_lock"


def get_deployment_lock(zookeeper_client: MarqoZookeeperClient, acquire_timeout=1) -> ZookeeperDistributedLock:
    """Get the deployment lock.

    Args:
        zookeeper_client: The Zookeeper client.
        acquire_timeout: The timeout to acquire the lock.

    Returns:
        AbstractDistributedLock: The deployment lock.
    """
    return ZookeeperDistributedLock(zookeeper_client, _DEPLOYMENT_LOCK_PATH, acquire_timeout)
