from .element_store import ElementStore, EmbeddedElement, Element
from ..module import ModuleConfiguration


class MockElementStore(ElementStore):
    __elements: list[EmbeddedElement] = []

    def __init__(self, configuration: ModuleConfiguration):
        pass

    def create_vector_store(self, previous_modules_key: str, entries: list[EmbeddedElement]):
        self.__elements = entries

    def find_similar(self, query: list[float]) -> list[Element]:
        return [element.element for element in self.__elements]

    def get_by_parent_id(self, identifier: str) -> list[EmbeddedElement]:
        return [element for element in self.__elements if element.element.parent.identifier == identifier]

    def get_all_elements(self, compare: bool = False) -> list[EmbeddedElement]:
        return self.__elements
