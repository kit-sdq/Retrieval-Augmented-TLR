from typing import NamedTuple
from typing import Protocol

from ..knowledge import Element
from ..module import ModuleConfiguration


class EmbeddedElement(NamedTuple):
    element: Element
    embedding: list[float]


class ElementStore(Protocol):

    def create_vector_store(self,
                            previous_modules_key: str,
                            entries: list[EmbeddedElement]):
        ...

    def find_similar(self, query: list[float]) -> list[Element]:
        ...

    def find_similar_with_distances(self, query: list[float]) -> (list[Element], list[float]):
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
    from .custom_element_store import CustomElementStore

    STORES = {
        'mock': MockElementStore,
        'chroma': ChromaElementStore,
        'custom': CustomElementStore
    }

    def build_element_store(self, configuration: ModuleConfiguration) -> ElementStore:
        return self.STORES[configuration.name](configuration)
