from kazoo.client import KazooClient
from kazoo.recipe.lock import Lock

import marqo.logging
from marqo.vespa.exceptions import ZookeeperTimeoutError

logger = marqo.logging.get_logger(__name__)


class DistributedLock:
    def __init__(self, lock: Lock):
        self.lock = lock

    def release(self):
        logger.debug(f"Releasing lock {self.lock.path}")
        self.lock.release()

    def is_acquired(self):
        return self.lock.is_acquired


class ZookeeperClient:
    _LOCK_VESPA_DEPLOYMENT = "/marqo__vespa_deployment_lock"

    def __init__(self, hosts: str):
        self.hosts = hosts
        self.client = KazooClient(hosts=self.hosts)

        logger.debug(f"Connecting to Zookeeper at {self.hosts}")
        self.client.start()
        logger.debug(f"Connected to Zookeeper at {self.hosts}")

    def close(self):
        self.client.stop()
        self.client.close()

    def lock_vespa_deployment(self, timeout: int = 10) -> DistributedLock:
        lock = Lock(self.client, self._LOCK_VESPA_DEPLOYMENT)

        logger.debug(f"Acquiring lock {self._LOCK_VESPA_DEPLOYMENT} with timeout {timeout}")

        if lock.acquire(timeout=timeout):
            return DistributedLock(lock)
        else:
            raise ZookeeperTimeoutError(f"Failed to acquire lock within {timeout} seconds")
