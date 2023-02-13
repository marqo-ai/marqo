from marqo.tensor_search.enums import ThrottleType
from marqo.connections import redis_driver
from marqo.tensor_search.enums import RequestType, EnvVars
from marqo.tensor_search import utils
from marqo.tensor_search.tensor_search_logging import get_logger
from marqo.errors import TooManyRequestsError
from functools import wraps
from threading import Thread
import uuid

# for logging
import datetime
import time
import os
import logging

logger = get_logger(__name__)

def throttle(request_type: str):
    """
    Decorator that checks if a user has exceeded their throttling limits.
    Throttling types:
    Current: thread_count
    For future implementation: data_size, per_user, etc.

    Implemented in a failsafe manner. If redis cannot be connected to or causes an error for any reason, this function is escaped and marqo operation will proceed as normal.
    Can be manually turned off with env var: $MARQO_ENABLE_THROTTLING='FALSE'
    """
    def decorator(function):
        
        @wraps(function)        # needed to preserve function metadata, or else FastAPI throws a 422.
        def wrapper(*args, **kwargs):

            if utils.read_env_vars_and_defaults(EnvVars.MARQO_ENABLE_THROTTLING) != "TRUE":
                return function(*args, **kwargs)

            redis = redis_driver.get_db()  # redis instance
            lua_shas = redis_driver.get_lua_shas()

            # Define maximum thread counts
            throttling_max_threads = {
                RequestType.INDEX: utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_CONCURRENT_INDEX),
                RequestType.SEARCH: utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_CONCURRENT_SEARCH) 
            }
            
            set_key = f"set:{request_type}"
            thread_name = f"thread:{uuid.uuid4()}"

            t0 = time.time()

            def remove_thread_from_set(key, name):
                try:
                    redis.zrem(key, name)
                except Exception as e:
                    logger.warn(f"There is a problem with your redis instance. Skipping throttling decrement. Reason: {e}")
                    redis_driver.set_faulty(True)

            # Check current thread count / increment using LUA script
            try:
                check_result = redis.evalsha(
                    lua_shas["check_and_increment"], 
                    1,          
                    set_key,                                 # sorted set key (by request type)
                    thread_name,                             # name of member for the thread
                    throttling_max_threads[request_type],    # thread_limit
                    utils.read_env_vars_and_defaults(EnvVars.MARQO_THREAD_EXPIRY_TIME)  # expire_time
                )
            except Exception as e:
                logger.warn(f"Could not load throttling scripts onto Redis. There is likely a problem with your redis instance or connection. Skipping throttling check. Reason: {e}")
                redis_driver.set_faulty(True)
                return function(*args, **kwargs)

            t1 = time.time()
            redis_time = (t1 - t0)*1000

            # Thread limit exceeded, throw 429
            if check_result != 0:
                throttling_message = f"Throttled because maximum thread count ({throttling_max_threads[request_type]}) for request type '{request_type}' has been exceeded. Try your request again later."
                raise TooManyRequestsError(message=throttling_message)

            else:
                # Execute function
                try:
                    result = function(*args, **kwargs)
                    return result

                except Exception as e:
                    raise e
                
                # Delete thread key whether function succeeds or fails (async)
                finally:
                    # Remove key from sorted set (async)
                    remove_thread = Thread(target = remove_thread_from_set, args = (set_key, thread_name))
                    remove_thread.start()
                    
        return wrapper
    return decorator