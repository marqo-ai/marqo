import redis
import os
from marqo import errors
from marqo.tensor_search.tensor_search_logging import get_logger
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars

# for logging
import datetime
import time
import os
import logging

"""
Drivers for connecting to other applications should be put here.
"""

logger = get_logger(__name__)

class RedisDriver:
    """
    This class enables a persistent connection to redis
    """


    def __init__(self):
        self.driver: redis.Redis = None
        self.faulty: bool = False
        self.lua_shas: dict = None
        self.host: str = "localhost"
        self.port: int = 6379

        # Specify any LUA scripts to be used by the driver here.
        self.scripts = [
            {
                "name": "check_and_increment",
                "path": "throttling/check_and_increment.lua"
            },
            # No need for exit script, as it's only 1 line
            #{
            #    "name": "exit_thread",
            #    "path": "tensor_search/throttling/exit_thread.lua"
            #},
        ]  

    def init_from_app(self, host, port):
        t0 = time.time()

        self.host = host
        self.port = port

        self.connect()

        try:
            self.load_lua_scripts()
            # remove any info from last Marqo instance
            self.driver.flushdb()
            self.faulty = False

            t1 = time.time()
            logger.info(f"Took {((t1-t0)*1000):.3f}ms to connect to redis and load scripts.")

        except Exception as e:
            logger.warn(f"There is a problem with your redis instance. Could not load scripts. Reason: {e}")
            self.faulty = True

    def connect(self) -> redis.Redis:
        # Does not raise an exception even when no redis-server is available. Monitor this in case that changes.
        self.driver = redis.Redis(
            host=self.host,
            port=self.port,
        )
        return self.driver
        
    def load_lua_scripts(self) -> dict:
        self.lua_shas = dict()
        for script in self.scripts:
            script_file = open(script["path"], "r")
            script_text = script_file.read()

            self.lua_shas[script["name"]] = self.driver.script_load(script_text)
            script_file.close()
        return self.lua_shas

    def get_db(self):
        # Try to connect if no good connection exists yet.
        if not self.driver or self.faulty:
            self.init_from_app(self.host, self.port)
        return self.driver
    
    def set_faulty(self, value):
        self.faulty = value
    
    def get_lua_shas(self):
        return self.lua_shas
    
    def is_alive(self):
        # Not using this, in the interest of speed. Pinging every connection adds latency.
        try: 
            self.driver.ping()
            return True
        except Exception as e:
            logger.warn(f"Could not confirm connection to redis. Ensure you have a redis server running for throttling. Reason: {e}")
            return False

# Starts up redis driver
# Can be turned off with MARQO_ENABLE_THROTTLING = 'FALSE'
if utils.read_env_vars_and_defaults(EnvVars.MARQO_ENABLE_THROTTLING):
    redis_driver = RedisDriver()