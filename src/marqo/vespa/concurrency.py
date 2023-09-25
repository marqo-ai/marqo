import asyncio
import concurrent


def _run_coroutine_new_event_loop(loop, coro):
    # asyncio.set_event_loop(loop)
    # return loop.run_until_complete(coro)
    asyncio.run(coro)


def run_coroutine(coro):
    try:
        _ = asyncio.get_running_loop()
        new_loop = asyncio.new_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                _run_coroutine_new_event_loop, new_loop, coro
            )
            return_value = future.result()
            return return_value
    except RuntimeError:
        return asyncio.run(coro)
