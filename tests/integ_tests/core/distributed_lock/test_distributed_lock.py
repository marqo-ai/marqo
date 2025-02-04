from unittest.mock import patch

from kazoo.exceptions import LockTimeout, ConnectionClosedError
from kazoo.handlers.threading import KazooTimeoutError

from marqo.core.distributed_lock.zookeeper_distributed_lock import get_deployment_lock
from marqo.core.distributed_lock.zookeeper_distributed_lock import ZookeeperDistributedLock
from marqo.core.exceptions import BackendCommunicationError
from integ_tests.marqo_test import MarqoTestCase
from marqo.core.exceptions import ZookeeperLockNotAcquiredError


class TestZookeeperDistributedLock(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        try:
            cls.zookeeper_client.start(5)
        except KazooTimeoutError as e:
            raise ConnectionError("Failed to connect to Zookeeper during the unit tests. "
                                  "Please ensure the Zookeeper is configured and published ") from e

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        # Restart the Zookeeper client after the test class
        cls.zookeeper_client.restart()

    def setUp(self):
        # Set up the Kazoo Test Harness
        self.zookeeper_client.start()
        self.acquire_timeout = 5  # 5 seconds to acquire the lock

    def tearDown(self):
        # Stop the client after each test to release the lock
        self.zookeeper_client.stop()

    def test_distributed_lock_acquireAndReleaseLock(self):
        """Test that the lock can be acquired and released properly."""
        lock = get_deployment_lock(self.zookeeper_client, self.acquire_timeout)
        self.assertTrue(lock.acquire())
        self.assertTrue(lock.is_acquired)
        lock.release()
        self.assertFalse(lock.is_acquired)

    def test_distributed_lock_concurrentLockAcquisition(self):
        """Test that a second lock cannot be acquired concurrently with the first lock."""
        lock1 = get_deployment_lock(self.zookeeper_client, self.acquire_timeout)
        lock2 = get_deployment_lock(self.zookeeper_client, self.acquire_timeout)

        self.assertTrue(lock1.acquire())
        with self.assertRaises(ZookeeperLockNotAcquiredError):
            lock2.acquire()

        lock1.release()
        self.assertTrue(lock2.acquire())
        lock2.release()
        self.assertFalse(lock2.is_acquired)
        self.assertFalse(lock1.is_acquired)

    def test_distributed_lock_lockAcquisitionTimeout(self):
        """Test lock acquisition fails when timeout is reached."""
        lock = get_deployment_lock(self.zookeeper_client, self.acquire_timeout)  # Short timeout for the test
        with self.assertRaises(ZookeeperLockNotAcquiredError):
            with patch.object(lock._lock, 'acquire', side_effect=LockTimeout):
                self.assertFalse(lock.acquire())

    def test_distributed_lock_repeatedAcquireRelease(self):
        """Test acquiring and releasing the lock repeatedly."""
        lock = get_deployment_lock(self.zookeeper_client, self.acquire_timeout)
        for _ in range(5):
            self.assertTrue(lock.acquire())
            self.assertTrue(lock.is_acquired)
            lock.release()
            self.assertFalse(lock.is_acquired)

    def test_distributed_lock_independentLocks(self):
        """Test that multiple locks on different paths do not interfere with each other."""
        lock1 = ZookeeperDistributedLock(self.zookeeper_client, "/lock1")
        lock2 = ZookeeperDistributedLock(self.zookeeper_client, "/lock2")
        self.assertTrue(lock1.acquire())
        self.assertTrue(lock2.acquire())
        lock1.release()
        lock2.release()

    def test_distributed_lock_zeroTimeoutAcquisition(self):
        """Test lock acquisition with zero timeout."""
        lock = get_deployment_lock(self.zookeeper_client, acquire_timeout=0)
        self.assertTrue(lock.acquire())

    def test_distributed_lock_handleKazooTimeoutErrorGracefully(self):
        """Test the context manager handles Kazoo timeout error close gracefully."""
        lock = get_deployment_lock(self.zookeeper_client, self.acquire_timeout)
        with patch.object(lock._zookeeper_client, 'state', 'LOST'):
            with patch.object(lock._zookeeper_client, 'start', side_effect=KazooTimeoutError):
                with self.assertRaises(BackendCommunicationError):
                    with lock:
                        pass
            self.assertFalse(lock.is_acquired)

    def test_distributed_lock_handleConnectionClosedGracefully(self):
        """Test the context manager handles connection closed error gracefully."""
        lock = get_deployment_lock(self.zookeeper_client, self.acquire_timeout)
        with patch.object(lock._lock, 'acquire', side_effect=ConnectionClosedError):
            with self.assertRaises(BackendCommunicationError):
                with lock:
                    pass
        self.assertFalse(lock.is_acquired)

    def test_the_same_distributed_lock_canNotBeAcquiredTwice(self):
        """Test that the same lock cannot be acquired twice."""
        lock = get_deployment_lock(self.zookeeper_client, self.acquire_timeout)
        self.assertTrue(lock.acquire())
        with self.assertRaises(ZookeeperLockNotAcquiredError):
            lock.acquire()