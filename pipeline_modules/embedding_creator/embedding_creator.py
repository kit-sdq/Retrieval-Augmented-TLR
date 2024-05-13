from typing import Protocol, Any
from ..knowledge import Element

from ..module import ModuleConfiguration


class Embedding:
    # depending on third party libraries there can be many different implementations of embeddings.
    # often they are list[float]
    embedding: Any

    def __init__(self, embedding: Any):
        self.embedding = embedding


class EmbeddingCreator(Protocol):
    def calculate_embedding(self, element: Element) -> Embedding:
        ...

    def calculate_multiple_embeddings(self, elements: list[Element]) -> list[Embedding]:
        ...


class EmbeddingCreatorBuilder:
    from .mock_embedding_creator import MockEmbeddingCreator
    from .openai_embedding_creator import OpenAIEmbeddingCreator

    EMBEDDING_CREATORS = {
        'mock': MockEmbeddingCreator,
        'open_ai': OpenAIEmbeddingCreator
    }

    def build_embedding_creator(self, configuration: ModuleConfiguration) -> EmbeddingCreator:
        return self.EMBEDDING_CREATORS[configuration.name](configuration)
