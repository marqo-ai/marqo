import time
from unittest.mock import patch

from kazoo.exceptions import LockTimeout, ConnectionClosedError
from kazoo.handlers.threading import KazooTimeoutError

from marqo.core.distributed_lock.distributed_lock import DistributedLock, acquire_lock
from marqo.core.exceptions import MarqoError
from tests.marqo_test import MarqoTestCase


class TestDistributedLock(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        try:
            cls.zookeeper_client.start()
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
        self.path = "/test-lock"
        self.max_lock_period = 10  # 10 seconds max lock period for testing
        self.watchdog_interval = 1  # 1 second watchdog check interval
        self.acquire_timeout = 5  # 5 seconds to acquire the lock

    def tearDown(self):
        # Stop the client after each test to release the lock
        self.zookeeper_client.stop()

    def test_distributed_lock_acquireAndReleaseLock(self):
        """Test that the lock can be acquired and released properly."""
        lock = DistributedLock(self.zookeeper_client, self.path, self.max_lock_period, self.watchdog_interval,
                               self.acquire_timeout)
        self.assertTrue(lock.acquire())
        self.assertTrue(lock.lock.is_acquired)
        lock.release()
        self.assertFalse(lock.lock.is_acquired)

    def test_distributed_lock_lockExpiration(self):
        """Test that the lock expires and is released by the watchdog after the maximum lock period."""
        lock = DistributedLock(self.zookeeper_client, self.path, 2, 1, self.acquire_timeout)
        self.assertTrue(lock.acquire())
        time.sleep(5)  # Wait for the lock to expire
        self.assertFalse(lock.lock.is_acquired)

    def test_distributed_lock_concurrentLockAcquisition(self):
        """Test that a second lock cannot be acquired concurrently with the first lock."""
        lock1 = DistributedLock(self.zookeeper_client, self.path, self.max_lock_period, self.watchdog_interval,
                                self.acquire_timeout)
        lock2 = DistributedLock(self.zookeeper_client, self.path, self.max_lock_period, self.watchdog_interval,
                                self.acquire_timeout)

        self.assertTrue(lock1.acquire())
        self.assertFalse(lock2.acquire(self.acquire_timeout))

        lock1.release()

    def test_distributed_lock_lockAcquisitionTimeout(self):
        """Test lock acquisition fails when timeout is reached."""
        lock = DistributedLock(self.zookeeper_client, self.path, self.max_lock_period, self.watchdog_interval,
                               1)  # Short timeout for the test
        self.assertTrue(lock.acquire())
        with patch.object(lock.lock, 'acquire', side_effect=LockTimeout):
            self.assertFalse(lock.acquire())

    def test_distributed_lock_contextManagerUsage(self):
        """Test the context manager for acquiring and releasing the lock."""
        lock = DistributedLock(self.zookeeper_client, self.path, self.max_lock_period, self.watchdog_interval,
                               self.acquire_timeout)
        self.assertTrue(lock.acquire())
        with self.assertRaises(MarqoError):
            with acquire_lock(lock, MarqoError("Failed to acquire lock"), self.acquire_timeout):
                self.assertTrue(lock.lock.is_acquired)

    def test_distributed_lock_repeatedAcquireRelease(self):
        """Test acquiring and releasing the lock repeatedly."""
        lock = DistributedLock(self.zookeeper_client, self.path, self.max_lock_period)
        for _ in range(5):
            self.assertTrue(lock.acquire())
            self.assertTrue(lock.lock.is_acquired)
            lock.release()
            self.assertFalse(lock.lock.is_acquired)

    def test_distributed_lock_independentLocks(self):
        """Test that multiple locks on different paths do not interfere with each other."""
        lock1 = DistributedLock(self.zookeeper_client, "/lock1", self.max_lock_period)
        lock2 = DistributedLock(self.zookeeper_client, "/lock2", self.max_lock_period)
        self.assertTrue(lock1.acquire())
        self.assertTrue(lock2.acquire())
        lock1.release()
        lock2.release()

    def test_distributed_lock_acquireAfterExpiration(self):
        """Test that a lock can be acquired immediately after it expires."""
        lock = DistributedLock(self.zookeeper_client, self.path, 2, 1, self.acquire_timeout)
        self.assertTrue(lock.acquire())
        time.sleep(5)  # Expire the lock
        self.assertFalse(lock.lock.is_acquired)
        self.assertTrue(lock.acquire())  # Re-acquire the lock
        lock.release()

    def test_distributed_lock_zeroTimeoutAcquisition(self):
        """Test lock acquisition with zero timeout."""
        lock = DistributedLock(self.zookeeper_client, self.path, self.max_lock_period,
                               self.watchdog_interval, 0)
        self.assertTrue(lock.acquire())

    def test_distributed_lock_lockTimeoutReturnsFalse(self):
        """Test that a LockTimeout error results in a return value of False rather than raising an exception."""
        lock = DistributedLock(self.zookeeper_client, self.path, self.max_lock_period, self.watchdog_interval,
                               self.acquire_timeout)
        with patch.object(lock.lock, 'acquire', side_effect=LockTimeout):
            self.assertFalse(lock.acquire())

    def test_distributed_lock_handleConnectionClosedGracefully(self):
        """Test the context manager handles Zookeeper connection close gracefully."""
        lock = DistributedLock(self.zookeeper_client, self.path, self.max_lock_period, self.watchdog_interval,
                               self.acquire_timeout)

        with patch.object(lock, 'acquire', side_effect=ConnectionClosedError), \
                patch('marqo.core.distributed_lock.distributed_lock.logger.warning') as mock_logger:
            # Use the context manager
            with acquire_lock(lock, MarqoError("Failed to acquire lock")):
                pass

        # Check if the warning was logged
        mock_logger.assert_called_once_with("Zookeeper connection closed when trying to acquire lock. "
                                            "Skipping lock acquisition and proceeding. "
                                            "Marqo may not be protected by Zookeeper. "
                                            "Please check your Zookeeper configuration and network settings.")

        # Ensure lock.release() was not called since acquire should have failed
        self.assertFalse(lock.lock.is_acquired)