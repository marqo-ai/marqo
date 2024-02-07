from kazoo.client import KazooClient
from kazoo.recipe.lock import Lock

from marqo.vespa.exceptions import ZookeeperTimeoutError


class DistributedLock:
    def __init__(self, lock: Lock):
        self.lock = lock

    def release(self):
        self.lock.release()

    def is_acquired(self):
        return self.lock.is_acquired


class ZookeeperClient:
    _LOCK_VESPA_DEPLOYMENT = "/marqo__vespa_deployment_lock"

    def __init__(self, hosts: str):
        self.hosts = hosts
        self.client = KazooClient(hosts=self.hosts)
        self.client.start()

    def close(self):
        self.client.stop()
        self.client.close()

    def lock_vespa_deployment(self, timeout: int = 10) -> DistributedLock:
        lock = Lock(self.client, self._LOCK_VESPA_DEPLOYMENT)
        if lock.acquire(timeout=timeout):
            return DistributedLock(lock)
        else:
            raise ZookeeperTimeoutError(f"Failed to acquire lock within {timeout} seconds")
