from typing import Protocol, Any

from ..knowledge import Element
from ..module import ModuleConfiguration



class EmbeddingCreator(Protocol):
    def calculate_embedding(self, element: Element) -> list[float]:
        ...

    def calculate_multiple_embeddings(self, elements: list[Element]) -> list[list[float]]:
        ...


class EmbeddingCreatorBuilder:
    from .mock_embedding_creator import MockEmbeddingCreator
    from .openai_embedding_creator import OpenAIEmbeddingCreator
    from .ollama_embedding_creator import OLLAMAEmbeddingCreator

    EMBEDDING_CREATORS = {
        'mock': MockEmbeddingCreator,
        'open_ai': OpenAIEmbeddingCreator,
        'ollama': OLLAMAEmbeddingCreator
    }

    def build_embedding_creator(self, configuration: ModuleConfiguration) -> EmbeddingCreator:
        return self.EMBEDDING_CREATORS[configuration.name](configuration)
