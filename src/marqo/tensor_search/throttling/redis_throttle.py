from marqo.tensor_search.enums import ThrottleType
from marqo.connections import redis_driver
from marqo.tensor_search.enums import RequestType, EnvVars
from marqo.tensor_search import utils
from marqo.errors import TooManyRequestsError
from functools import wraps
import uuid

# for logging
import datetime
import time
import os
import logging

def get_logger(name):
    test_throttle_timing_file = 'test_throttle_timing.txt'
    throttle_handler = logging.FileHandler(filename=test_throttle_timing_file)
    throttle_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s \n%(message)s"))
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(throttle_handler)

    return logger

def get_logger_parse(name):
    test_throttle_timing_file = 'test_throttle_timing.csv'
    throttle_handler = logging.FileHandler(filename=test_throttle_timing_file)
    throttle_handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger(name + "_parsed")
    logger.setLevel(logging.INFO)
    logger.addHandler(throttle_handler)

    return logger

logger = get_logger(__name__)
logger_parse = get_logger_parse(__name__)


def throttle(request_type: str):
    """
    Decorator that checks if a user has exceeded their throttling limits.
    """
    def decorator(function):
        
        @wraps(function)        # needed to preserve function metadata, or else FastAPI throws a 422.
        def wrapper(*args, **kwargs):
            print(f"Beginning throttling process. Your request is {request_type}")
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

            # Check current thread count / increment using LUA script
            check_result = redis.evalsha(
                lua_shas["check_and_increment"], 
                1,          
                set_key,                                 # sorted set key (by request type)
                thread_name,                             # name of member for the thread
                throttling_max_threads[request_type],    # thread_limit
                utils.read_env_vars_and_defaults(EnvVars.MARQO_THREAD_EXPIRY_TIME)  # expire_time
            )

            # put entire verbose log here.
            t1 = time.time()
            redis_time = (t1 - t0)*1000

            log_msg = f"THROTTLING CHECK\n"
            log_msg += f"Thread Name: {thread_name}\n"
            log_msg += f"Redis Check Time: {((t1 - t0)*1000):.3f}ms\n" 
            result_word = "PASSED" if check_result != 0 else "FAILED"
            log_msg += f"Result: {result_word} \n"
            log_msg += f"Set Key: {set_key}\n"
            log_msg += f"Max Threads: {throttling_max_threads[request_type]}\n"
            log_msg += f"Expiry Time: {utils.read_env_vars_and_defaults(EnvVars.MARQO_THREAD_EXPIRY_TIME)}\n\n"

            #logger.info(msg=log_msg)

            # JSON test data log
            json_msg = '{"timestamp": "' + time.asctime() + '",'
            json_msg += '"action": "check",'
            json_msg += '"thread_name": "' + thread_name + '",'
            json_msg += '"result": "' + result_word + '",'
            json_msg += '"max_threads": ' + str(throttling_max_threads[request_type]) + ','
            json_msg += '"redis_time": ' + str(redis_time)
            json_msg += '}'
            #logger_parse.info(msg=json_msg)

            check_test_data = {
                "timestamp": time.asctime(),
                "action": "check",
                "thread_name": thread_name,
                "result": result_word,
                "max_threads": throttling_max_threads[request_type],
                "redis_time": redis_time
            }

            # Thread limit exceeded, throw 429
            if check_result == 0:
                throttling_message = f"Throttled because maximum thread count ({throttling_max_threads[request_type]}) for request type '{request_type}' has been exceeded. Try your request again later."
                raise TooManyRequestsError(message=throttling_message)

            else:
                # Execute function
                try:
                    result = function(*args, **kwargs)
                    result["check_test_data"] = check_test_data
                    return result
                except Exception as e:
                    # TODO: maybe do something more robust here
                    pass
                
                # Delete thread key whether function succeeds or fails
                finally:
                    
                    t0 = time.time()
                    # Remove key from sorted set
                    redis.zrem(set_key, thread_name)

                    # put entire verbose log here.
                    t1 = time.time()
                    log_msg = f"THREAD FINISHED. DELETING REDIS SET ELEMENT\n"
                    log_msg += f"Thread Name: {thread_name}\n"
                    log_msg += f"Redis Delete Time: {((t1 - t0)*1000):.3f}ms\n" 
                    log_msg += f"Set Key: {set_key}\n"
                    logger.info(msg=log_msg)
                    logger_parse.info(msg=f"delete,{thread_name},{((t1 - t0)*1000):.3f}")

        return wrapper
    return decorator