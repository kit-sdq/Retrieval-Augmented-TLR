from .embedding_creator import EmbeddingCreator, Embedding, Element, ModuleConfiguration


class MockEmbeddingCreator(EmbeddingCreator):

    def __init__(self, configuration: ModuleConfiguration):
        pass

    def calculate_embedding(self, element: Element) -> Embedding:
        return Embedding(embedding=[0.0])

    def calculate_multiple_embeddings(self, elements: list[Element]) -> list[Embedding]:
        return [Embedding(embedding=0.0) for element in elements]
