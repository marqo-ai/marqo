"""Unit tests for thread_diagnostics module."""
import io
import os
import signal
import sys
import threading
from unittest import mock

import pytest

from marqo.core.monitoring import thread_diagnostics


class TestSetupSignalHandlers:
    """Tests for setup_signal_handlers function."""

    @pytest.mark.skipif(os.name == "nt", reason="Signal handlers not available on Windows")
    def test_setup_signal_handlers_registers_sigusr1_and_sigusr2(self):
        """Test that signal handlers are registered for SIGUSR1 and SIGUSR2."""
        with mock.patch("faulthandler.enable") as mock_enable, \
             mock.patch("signal.signal") as mock_signal:
            thread_diagnostics.setup_signal_handlers()

            mock_enable.assert_called_once()
            # SIGUSR1 uses custom handler that prints PID header before faulthandler dump
            # SIGUSR2 uses Python-level thread dump
            assert mock_signal.call_count == 2
            mock_signal.assert_any_call(
                signal.SIGUSR1,
                thread_diagnostics._dump_traceback_with_pid
            )
            mock_signal.assert_any_call(
                signal.SIGUSR2,
                thread_diagnostics._dump_threads_to_stderr
            )

    @pytest.mark.skipif(os.name != "nt", reason="Only run on Windows")
    def test_setup_signal_handlers_skips_on_windows(self):
        """Test that signal handlers are not registered on Windows."""
        with mock.patch("faulthandler.enable") as mock_enable, \
             mock.patch("signal.signal") as mock_signal:
            thread_diagnostics.setup_signal_handlers()

            mock_enable.assert_not_called()
            mock_signal.assert_not_called()


class TestDumpThreadsToStderr:
    """Tests for _dump_threads_to_stderr function."""

    def test_dump_threads_to_stderr_writes_output(self):
        """Test that thread dump is written to stderr."""
        captured_stderr = io.StringIO()
        with mock.patch.object(sys, 'stderr', captured_stderr):
            thread_diagnostics._dump_threads_to_stderr(signal.SIGUSR2, None)

        output = captured_stderr.getvalue()
        assert "=== Python Thread Dump (SIGUSR2)" in output
        assert f"PID={os.getpid()}" in output
        assert f"=== End Thread Dump PID={os.getpid()} ===" in output
        # Should include the main thread
        assert "MainThread" in output

    def test_dump_threads_to_stderr_includes_thread_names(self):
        """Test that thread names are included in the dump."""
        test_thread_name = "TestDumpThread"
        barrier = threading.Barrier(2)

        def thread_target():
            barrier.wait()  # Wait for main thread to signal
            barrier.wait()  # Wait for main thread to complete dump

        thread = threading.Thread(target=thread_target, name=test_thread_name)
        thread.start()

        try:
            barrier.wait()  # Signal thread is ready
            captured_stderr = io.StringIO()
            with mock.patch.object(sys, 'stderr', captured_stderr):
                thread_diagnostics._dump_threads_to_stderr(signal.SIGUSR2, None)
            barrier.wait()  # Signal thread can exit

            output = captured_stderr.getvalue()
            assert test_thread_name in output
        finally:
            thread.join(timeout=1)


class TestDumpTracebackWithPid:
    """Tests for _dump_traceback_with_pid function."""

    def test_dump_traceback_with_pid_includes_pid_header_and_end_delimiter(self):
        """Test that PID header and END delimiter are included in faulthandler output."""
        captured_stderr = io.StringIO()
        with mock.patch.object(sys, 'stderr', captured_stderr), \
             mock.patch("faulthandler.dump_traceback") as mock_dump:
            thread_diagnostics._dump_traceback_with_pid(signal.SIGUSR1, None)

        output = captured_stderr.getvalue()
        pid = os.getpid()
        assert f"=== Faulthandler Thread Dump (SIGUSR1) PID={pid} ===" in output
        assert f"=== End Thread Dump PID={pid} ===" in output
        mock_dump.assert_called_once()


class TestGetThreadDump:
    """Tests for get_thread_dump function."""

    def test_get_thread_dump_returns_current_threads(self):
        """Test that get_thread_dump returns info about current threads."""
        dump = thread_diagnostics.get_thread_dump()

        assert "pid" in dump
        assert dump["pid"] == os.getpid()
        assert "thread_count" in dump
        assert dump["thread_count"] > 0
        assert "threads" in dump
        assert len(dump["threads"]) == dump["thread_count"]

    def test_get_thread_dump_thread_structure(self):
        """Test that each thread in the dump has the expected structure."""
        dump = thread_diagnostics.get_thread_dump()

        for thread_info in dump["threads"]:
            assert "id" in thread_info
            assert "name" in thread_info
            assert "stack" in thread_info
            assert isinstance(thread_info["stack"], list)

    def test_get_thread_dump_includes_thread_pool_stats(self):
        """Test that thread pool stats are included in the dump."""
        dump = thread_diagnostics.get_thread_dump()

        assert "thread_pool" in dump
        # Outside async context, should return error
        assert "error" in dump["thread_pool"]


class TestGetThreadPoolStats:
    """Tests for get_thread_pool_stats function."""

    def test_get_thread_pool_stats_returns_error_outside_async_context(self):
        """Test that get_thread_pool_stats returns error when not in async context."""
        stats = thread_diagnostics.get_thread_pool_stats()
        assert "error" in stats

    @pytest.mark.asyncio
    async def test_get_thread_pool_stats_returns_stats_in_async_context(self):
        """Test that get_thread_pool_stats returns proper stats in async context."""
        stats = thread_diagnostics.get_thread_pool_stats()

        assert "error" not in stats
        assert "active_threads" in stats
        assert "capacity" in stats
        assert "available_threads" in stats
        assert "waiting_tasks" in stats
        assert stats["capacity"] > 0
        assert stats["available_threads"] == stats["capacity"] - stats["active_threads"]
