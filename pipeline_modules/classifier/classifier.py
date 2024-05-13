from typing import Protocol

from .context_provider import ContextProvider
from ..knowledge import Element
from ..module import ModuleConfiguration


class ClassificationResult:
    source: Element
    target: list[Element]

    def __init__(self, source: Element, targets: list[Element]):
        self.source = source
        self.target = targets


class Classifier(Protocol):
    def classify(self, source: Element, targets: list[Element]) -> ClassificationResult:
        ...


class ClassifierBuilder:
    from .mock_classifier import MockClassifier
    from .simple_classifier import SimpleClassifier
    from .reasoning_classifier import ReasoningClassifier
    from .selection_classifier import SelectionClassifier
    from .multi_step_classifier import MultiStepClassifier
    from .simple_classifier_ollama import SimpleOllamaClassifier

    CLASSIFIERS = {
        'mock': MockClassifier,
        'simple': SimpleClassifier,
        'chain_of_thought': ReasoningClassifier,
        'selection': SelectionClassifier,
        'multi_step': MultiStepClassifier,
        'simple_ollama': SimpleOllamaClassifier
    }

    def build_classifier(self, configuration: ModuleConfiguration, context_provider: ContextProvider) -> Classifier:
        return self.CLASSIFIERS[configuration.name](configuration, context_provider)
