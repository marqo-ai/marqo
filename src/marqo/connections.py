import redis
import os
from marqo import errors

"""
Drivers for connecting to other applications should be put here.
"""

class RedisDriver:
    """
    This class enables a persistent connection to redis
    """


    def __init__(self):
        self.driver: redis.Redis = None
        self.lua_shas: dict = None

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
        self.connect(host, port)
        self.load_lua_scripts()
        
        # remove any info from last Marqo instance
        self.driver.flushdb()

    def connect(self, host, port) -> redis.Redis:
        self.driver = redis.Redis(
            host=host,
            port=port,
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
        if not self.driver:
            return self.connect()
        return self.driver

# Starts up redis driver
redis_driver = RedisDriver()