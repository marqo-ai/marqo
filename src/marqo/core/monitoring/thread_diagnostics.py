"""Thread diagnostics for observability and debugging.

This module provides signal handlers and utilities for diagnosing thread pool
starvation and other threading issues in production environments.

Signal Handlers:
    SIGUSR1: Uses faulthandler to dump thread tracebacks (C-level, works even if GIL is stuck)
    SIGUSR2: Python-level thread dump with full stack traces

Usage:
    # In application startup:
    from marqo.core.monitoring.thread_diagnostics import setup_signal_handlers
    setup_signal_handlers()

    # From command line:
    kill -USR1 <pid>  # C-level thread dump
    kill -USR2 <pid>  # Python-level thread dump
"""
import faulthandler
import logging
import os
import signal
import sys
import threading
import traceback
from typing import Any, Dict, List

import anyio

logger = logging.getLogger(__name__)


def _dump_threads_to_stderr(signum, frame):
    """SIGUSR2 handler: dump all thread stacks to stderr."""
    pid = os.getpid()
    thread_names = {t.ident: t.name for t in threading.enumerate()}
    lines = [f"\n=== Python Thread Dump (SIGUSR2) PID={pid} ===\n"]
    for thread_id, stack in sys._current_frames().items():
        name = thread_names.get(thread_id, f"Thread-{thread_id}")
        lines.append(f"\n--- {name} (id={thread_id}) ---\n")
        lines.extend(traceback.format_stack(stack))
    lines.append(f"=== End Thread Dump PID={pid} ===\n\n")
    sys.stderr.write("".join(lines))
    sys.stderr.flush()


def _dump_traceback_with_pid(signum, frame):
    """SIGUSR1 handler: dump tracebacks with PID header for worker identification."""
    pid = os.getpid()
    sys.stderr.write(f"\n=== Faulthandler Thread Dump (SIGUSR1) PID={pid} ===\n")
    sys.stderr.flush()
    faulthandler.dump_traceback()
    sys.stderr.write(f"=== End Thread Dump PID={pid} ===\n\n")
    sys.stderr.flush()


def setup_signal_handlers():
    """Register signal handlers for thread diagnostics.

    SIGUSR1: faulthandler dumps all thread tracebacks with PID header (C-level, works if GIL stuck)
    SIGUSR2: Python-level thread dump with full stack traces

    Note: These signals are only available on Unix-like systems.
    """
    if os.name != "nt":
        faulthandler.enable()
        signal.signal(signal.SIGUSR1, _dump_traceback_with_pid)
        signal.signal(signal.SIGUSR2, _dump_threads_to_stderr)
        logger.info("Thread dump signal handlers registered (SIGUSR1=faulthandler, SIGUSR2=python)")


def get_thread_pool_stats() -> Dict[str, Any]:
    """Get AnyIO thread pool limiter statistics.

    Must be called from within an async context (i.e., from the event loop thread).

    Returns:
        Dictionary with thread pool statistics:
        - active_threads: Number of threads currently in use
        - capacity: Total thread pool capacity
        - available_threads: Number of threads available
        - waiting_tasks: Number of tasks waiting for a thread
    """
    try:
        limiter = anyio.to_thread.current_default_thread_limiter()
        stats = limiter.statistics()
        return {
            "active_threads": stats.borrowed_tokens,
            "capacity": stats.total_tokens,
            "available_threads": stats.total_tokens - stats.borrowed_tokens,
            "waiting_tasks": stats.tasks_waiting,
        }
    except Exception as e:
        return {"error": str(e)}


def get_thread_dump() -> Dict[str, Any]:
    """Get thread dump information for the debug endpoint.

    Returns:
        Dictionary containing:
        - pid: Process ID
        - thread_count: Number of threads
        - thread_pool: Thread pool statistics (if called from async context)
        - threads: List of thread info with id, name, and stack trace
    """
    thread_names = {t.ident: t.name for t in threading.enumerate()}
    threads: List[Dict[str, Any]] = []

    for thread_id, stack in sys._current_frames().items():
        name = thread_names.get(thread_id, f"Thread-{thread_id}")
        frames = traceback.format_stack(stack)
        threads.append({
            "id": thread_id,
            "name": name,
            "stack": frames,
        })

    return {
        "pid": os.getpid(),
        "thread_count": len(threads),
        "thread_pool": get_thread_pool_stats(),
        "threads": threads,
    }
