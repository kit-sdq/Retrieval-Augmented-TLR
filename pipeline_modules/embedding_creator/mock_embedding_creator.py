from .embedding_creator import EmbeddingCreator, Element, ModuleConfiguration


class MockEmbeddingCreator(EmbeddingCreator):

    def __init__(self, configuration: ModuleConfiguration):
        pass

    def calculate_embedding(self, element: Element) -> list[float]:
        return [0.0]

    def calculate_multiple_embeddings(self, elements: list[Element]) -> list[list[float]]:
        return [[0.0] for element in elements]
