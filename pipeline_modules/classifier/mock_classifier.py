from .classifier import Classifier, Element, ClassificationResult
from .context_provider import ContextProvider
from ..element_store.element_store import ElementStore

from ..module import ModuleConfiguration


class MockClassifier(Classifier):
    def __init__(self, configuration: ModuleConfiguration, context_provider: ContextProvider):
        pass

    def classify(self, source: Element, targets: list[Element]) -> ClassificationResult:
        return ClassificationResult(source=source, targets=targets)
    