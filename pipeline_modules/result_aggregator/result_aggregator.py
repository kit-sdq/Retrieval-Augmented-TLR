from typing import Protocol

from ..classifier.classifier import ClassificationResult
from ..module import ModuleConfiguration


class TraceLink:
    source: str
    target: str

    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target

    def __eq__(self, other):
        if isinstance(other, TraceLink):
            return self.source == other.source and self.target == other.target
        return False

    def __hash__(self):
        return hash((self.source, self.target))


class ResultAggregator(Protocol):

    def aggregate(self, results: list[ClassificationResult]) -> set[TraceLink]:
        ...


class ResultAggregatorBuilder:
    from .any_result_aggregator import AnyResultAggregator

    AGGREGATORS = {
        'any_connection': AnyResultAggregator
    }

    def build_result_aggregator(self, configuration: ModuleConfiguration) -> ResultAggregator:
        return self.AGGREGATORS[configuration.name](configuration)