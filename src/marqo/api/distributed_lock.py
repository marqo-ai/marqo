from kazoo.client import KazooClient
from kazoo.retry import KazooRetry
from kazoo.exceptions import NodeExistsError
from kazoo.exceptions import NodeExistsError, NoNodeError
import threading



class DistributedLock:
    def __init__(self, zk_hosts, lock_path):
        self.zk = KazooClient(hosts=zk_hosts)
        self.zk.start()
        self.lock_path = lock_path
        self.lock_node = None
        self.lock_acquired_event = threading.Event()

        # Ensure the lock path exists
        self._ensure_path()

    def _ensure_path(self):
        try:
            self.zk.ensure_path(self.lock_path)
        except NoNodeError:
            # Create the parent path if it does not exist
            self.zk.ensure_path(self.lock_path)

    def _watcher_callback(self, event):
        # Trigger the event when the znode we are watching is deleted
        if event.type == "DELETED":
            self.lock_acquired_event.set()

    def acquire(self):
        retry = KazooRetry(max_tries=3)

        while True:
            try:
                # Create an ephemeral sequential znode
                self.lock_node = self.zk.create(
                    self.lock_path + "/lock-",
                    ephemeral=True,
                    sequence=True
                )
                break
            except NodeExistsError:
                continue

        while True:
            # Get the list of znodes in the lock path
            children = self.zk.get_children(self.lock_path)
            children.sort()

            # Check if the created znode has the lowest sequence number
            if self.lock_node.split("/")[-1] == children[0]:
                return True

            # Watch the znode with the next lowest sequence number
            index = children.index(self.lock_node.split("/")[-1]) - 1
            predecessor = children[index]
            predecessor_path = self.lock_path + "/" + predecessor

            self.lock_acquired_event.clear()
            self.zk.exists(predecessor_path, watch=self._watcher_callback)
            self.lock_acquired_event.wait()

    def release(self):
        if self.lock_node:
            self.zk.delete(self.lock_node)
            self.lock_node = None

    def close(self):
        self.zk.stop()