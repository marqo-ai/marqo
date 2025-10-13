from kazoo.client import KazooClient


class ZookeeperClient(KazooClient):
    """A wrapper around the KazooClient to provide a timeout for the start method."""

    def __init__(self, zookeeper_connection_timeout: float, **kwargs):
        super().__init__(**kwargs)
        self.zookeeper_connection_timeout = zookeeper_connection_timeout

    def start(self, timeout: float = None):
        super().start(timeout if timeout is not None else self.zookeeper_connection_timeout)
