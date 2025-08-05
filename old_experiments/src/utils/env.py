from os import getenv as env, path
from dotenv import load_dotenv
from old_experiments.src.utils.singleton import singleton


@singleton
class Env:
    def __init__(self):
        load_dotenv(
            dotenv_path=path.join(path.dirname(path.realpath(__file__)),
                                  "../..", ".env")
        )
        self.__env: dict[str, str | None] = {}

    def __getitem__(self, key: str) -> str or None:
        if key not in self.__env:
            self.__env[key] = env(key)
        return self.__env[key]