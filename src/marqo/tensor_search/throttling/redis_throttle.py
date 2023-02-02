from marqo.tensor_search.enums import ThrottleType
from marqo.connections import redis_driver

def throttle(request_type: str):
    """
    Decorator that checks if a user has exceeded their throttling limits.
    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            print(f"Beginning throttling process. Your request is {request_type}")
            redis = redis_driver.get_db()  # redis instance

            # Define maximum thread counts
            throttling_max_threads = {
                RequestType.INDEX: utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_CONCURRENT_INDEX),
                RequestType.SEARCH: utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_CONCURRENT_SEARCH) 
            }
            
            set_key = f"set:{request_type}"
            thread_name = f"thread:{uuid.uuid4()}"

            print(f"ABOUT TO CHECK REDIS FOR {set_key} WITH THREAD NAME {thread_name}")
            # Check current thread count / increment using LUA script
            check_result = redis.evalsha(
                redis.lua_sha["check_and_increment"], 
                1,          
                set_key,                                 # sorted set key (by request type)
                thread_name,                             # name of member for the thread
                throttling_max_threads[request_type],    # thread_limit
                utils.read_env_vars_and_defaults(EnvVars.MARQO_THREAD_EXPIRY_TIME)  # expire_time
            )

            # Thread limit exceeded, throw 429
            if check_result == 0:
                throttling_message = f"Throttled because maximum thread count ({throttling_max_threads[request_type]}) for request type '{request_type}' "
                f"has been exceeded. Try your request again later."
                raise TooManyRequestsError(message=throttling_message)

            else:
                # Execute function
                try:
                    print(f"THREAD CREATED IN SET {set_key} WITH NAME {thread_name}")
                    result = function(*args, **kwargs)
                    return result
                except Exception and e:
                    # TODO: maybe do something more robust here
                    pass
                
                # Delete thread key whether function succeeds or fails
                finally:
                    print(f"THREAD WITH NAME {thread_name} FINISHED. REMOVING FROM SET {set_key}")
                    # Remove key from sorted set
                    redis.zrem(set_key, thread_name)

        return wrapper
    return decorator
