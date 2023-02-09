from marqo.tensor_search.enums import ThrottleType
from marqo.connections import redis_driver
from marqo.tensor_search.enums import RequestType, EnvVars
from marqo.tensor_search import utils
from marqo.errors import TooManyRequestsError
from functools import wraps
from threading import Thread
import uuid

# for logging
import datetime
import time
import os
import logging

def throttle(request_type: str):
    """
    Decorator that checks if a user has exceeded their throttling limits.
    """
    def decorator(function):
        
        @wraps(function)        # needed to preserve function metadata, or else FastAPI throws a 422.
        def wrapper(*args, **kwargs):

            # Can be turned off with MARQO_ENABLE_THROTTLING = 'FALSE'
            if utils.read_env_vars_and_defaults(EnvVars.MARQO_ENABLE_THROTTLING) != "TRUE":
                return function(*args, **kwargs)

            redis = redis_driver.get_db()  # redis instance
            
            # Escape throttle if redis connection is problematic
            if not redis:
                return function(*args, **kwargs)

            print(f"Beginning throttling process. API request is of type {request_type}")
            lua_shas = redis_driver.get_lua_shas()

            # Define maximum thread counts
            throttling_max_threads = {
                RequestType.INDEX: utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_CONCURRENT_INDEX),
                RequestType.SEARCH: utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_CONCURRENT_SEARCH) 
            }
            
            set_key = f"set:{request_type}"
            thread_name = f"thread:{uuid.uuid4()}"

            t0 = time.time()

            # Check current thread count / increment using LUA script
            # try ping, except escape
            check_result = redis.evalsha(
                lua_shas["check_and_increment"], 
                1,          
                set_key,                                 # sorted set key (by request type)
                thread_name,                             # name of member for the thread
                throttling_max_threads[request_type],    # thread_limit
                utils.read_env_vars_and_defaults(EnvVars.MARQO_THREAD_EXPIRY_TIME)  # expire_time
            )

            t1 = time.time()
            redis_time = (t1 - t0)*1000
            
            """
            Only for testing
            check_test_data = {
                "timestamp": time.asctime(),
                "action": "check",
                "thread_name": thread_name,
                "result": result_word,
                "max_threads": throttling_max_threads[request_type],
                "redis_time": redis_time
            }
            """

            # Thread limit exceeded, throw 429
            if check_result == 0:
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

                    def remove_thread_from_set(key, name):
                        # try ping, except escape
                        redis.zrem(key, name)
                    
                    # Remove key from sorted set (async)
                    remove_thread = Thread(target = remove_thread_from_set, args = (set_key, thread_name))
                    remove_thread.start()
                    
        return wrapper
    return decorator