import asyncio
import concurrent


def _run_coroutine_in_thread(coro):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(
            asyncio.run, coro
        )
        return_value = future.result()
        return return_value


def run_coroutine(coro):
    try:
        _ = asyncio.get_running_loop()
        # If no error, there's an existing loop in this thread, so we use a new thread
        return _run_coroutine_in_thread(coro)
    except RuntimeError:
        return asyncio.run(coro)
