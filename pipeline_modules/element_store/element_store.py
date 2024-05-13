from typing import NamedTuple
from typing import Protocol

from ..embedding_creator.embedding_creator import Embedding
from ..knowledge import Element
from ..module import ModuleConfiguration


class EmbeddedElement(NamedTuple):
    element: Element
    embedding: Embedding


class ElementStore(Protocol):

    def create_vector_store(self,
                            previous_modules_key: str,
                            entries: list[EmbeddedElement]):
        ...

    def find_similar(self, query: Embedding) -> list[Element]:
        ...

    def find_similar_with_distances(self, query: Embedding) -> (list[Element], list[float]):
        ...

    def get_by_id(self, identifier: str) -> Element:
        ...

    def get_by_parent_id(self, identifier: str) -> list[EmbeddedElement]:
        ...

    def get_all_elements(self, compare: bool = False) -> list[EmbeddedElement]:
        ...


class ElementStoreBuilder:
    from .mock_element_store import MockElementStore
    from .chroma_element_store import ChromaElementStore

    STORES = {
        'mock': MockElementStore,
        'chroma': ChromaElementStore
    }

    def build_element_store(self, configuration: ModuleConfiguration) -> ElementStore:
        return self.STORES[configuration.name](configuration)
