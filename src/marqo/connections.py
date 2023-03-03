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
            logger.warn(f"There is a problem with your redis connection. Could not load throttling scripts onto redis. To suppress these warnings, disable throttling with export MARQO_ENABLE_THROTTLING='FALSE'. Read more under Redis setup section of the developer guide: https://github.com/marqo-ai/marqo/tree/mainline/src/marqo#developer-guide. Redis error reason: {e}")
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

# Starts up redis driver
redis_driver = RedisDriver()