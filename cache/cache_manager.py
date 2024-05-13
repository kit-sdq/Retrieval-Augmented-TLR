import hashlib
import json
import os
import sqlite3

from pipeline_modules.module import ModuleConfiguration

cache: 'CacheManager | None'  # TODO: make actual Singleton or even better: find a good solution


class CacheManager:
    __folder_path: str
    __database_name: str

    __connection: sqlite3.Connection
    __cursor: sqlite3.Cursor

    def __init__(self, database_name: str, folder_path: str = "./storage"):
        self.__database_name = database_name
        self.__folder_path = folder_path
        if not os.path.exists(self.__folder_path):
            os.makedirs(self.__folder_path)
        self.__connection = sqlite3.connect(self.__folder_path + "/" + self.__database_name + ".sqlite3")
        self.__cursor = self.__connection.cursor()

        # Init table
        self.__create_cache_table()

        global cache
        cache = self

    def __create_cache_table(self):
        self.__cursor.execute('''CREATE TABLE IF NOT EXISTS cache(
                                        module TEXT,
                                        name TEXT, 
                                        config_hash TEXT, 
                                        config TEXT,
                                        input_hash TEXT, 
                                        input TEXT,
                                        data JSON,
                                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        self.__connection.commit()

    def __hash(self, data: str) -> str:
        sha256 = hashlib.sha256()
        sha256.update(data.encode())
        return sha256.hexdigest()

    def put(self, configuration: ModuleConfiguration, input: str, data: dict):
        parameters = {
            "module": configuration.type,
            "name": configuration.name,
            "config_hash": self.__hash(json.dumps(configuration.args, sort_keys=True)),
            "config": json.dumps(configuration.args, sort_keys=True),
            "input_hash": self.__hash(input),
            "input": input,
            "data": json.dumps(data, sort_keys=True),
        }
        self.__cursor.execute("INSERT INTO cache (module, name, config_hash, config, input_hash, input, data) "
                              "VALUES (:module, :name, :config_hash, :config, :input_hash, :input, json(:data))",
                              parameters)
        self.__connection.commit()

    def get(self, configuration: ModuleConfiguration, input_key: str) -> list[dict]:
        parameters = {
            "module": configuration.type,
            "name": configuration.name,
            "config_hash": self.__hash(json.dumps(configuration.args, sort_keys=True)),
            "input_hash": self.__hash(input_key),
        }
        data = self.__cursor.execute("SELECT data FROM cache WHERE module=:module AND name=:name AND config_hash=:config_hash AND input_hash=:input_hash", parameters)

        result: list[dict] = list()
        for row in data:
            result.append(json.loads(row[0]))
        return result

    @classmethod
    def get_cache(cls) -> 'CacheManager':
        global cache
        return cache
