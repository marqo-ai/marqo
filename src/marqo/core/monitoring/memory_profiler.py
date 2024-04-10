import tracemalloc

from memory_profiler import memory_usage

from marqo.core.models.memory_profile import MemoryProfile


def get_memory_profile() -> MemoryProfile:
    tracemalloc.start()

    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics('lineno')

    # Get mem used
    mem_used = memory_usage(-1, interval=0.1, timeout=1)

    return MemoryProfile(
        memory_used=mem_used[0],
        stats=[str(s) for s in stats]
    )
