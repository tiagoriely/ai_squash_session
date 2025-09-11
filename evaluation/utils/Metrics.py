# evaluation/utils/Metrics.py
from abc import abstractmethod
from typing import Any, Dict

class Metrics:
    def __init__(self):
        self.name = 'Metric'

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    @abstractmethod
    def get_score(self, **kwargs: Any) -> Any:
        pass