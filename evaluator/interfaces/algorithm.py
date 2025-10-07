from abc import ABC, abstractmethod
from typing import Dict, Any, ClassVar, List, Tuple, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from evaluator.components.data_provider import QuerySpecification
from evaluator.components.llm_provider import get_llm
from evaluator.config.schema import ModelConfig

# the algorithm result consists of the query response and the list of retrieved tools, if available
AlgoResponse = Tuple[Dict[str, Any], Union[List[str], None]]


class Algorithm(ABC):
    __algo_name__: ClassVar[str | None] = None

    def __init__(self, settings: Dict, model_config: List[ModelConfig], label: str = None):
        self._settings = settings
        defaults = self.get_default_settings()
        for key, value in defaults.items():
            self._settings.setdefault(key, value)

        self._label = label
        self._model_config = model_config

    @classmethod
    def get_name(cls) -> str:
        return cls.__algo_name__

    def get_unique_id(self) -> str:
        return f"{self.get_name()}:{self._settings}" if self._settings else self.get_name()

    def __str__(self) -> str:
        if self._label is not None:
            return self._label
        return self.get_unique_id()

    def _get_llm(self, model_id: str, **kwargs):
        """ A helper method for subclasses. """
        return get_llm(model_id, self._model_config, **kwargs)

    @abstractmethod
    def set_up(self, model: BaseChatModel, tools: List[BaseTool]) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def process_query(self, query_spec: QuerySpecification) -> AlgoResponse:
        raise NotImplementedError()

    @abstractmethod
    def tear_down(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_default_settings(self) -> Dict[str, Any]:
        raise NotImplementedError()
