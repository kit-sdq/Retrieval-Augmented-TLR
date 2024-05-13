from .preprocessor import Preprocessor
from ..knowledge import Artifact, Element
from ..module import ModuleConfiguration

from project.cache.cache_manager import CacheManager


class SimpleTextPreprocessor(Preprocessor):
    """A preprocessor using the original artifact text as content for a single Element."""
    __configuration: ModuleConfiguration

    def __init__(self, configuration: ModuleConfiguration):
        self.__configuration = configuration
        pass

    def __get_cached(self, artifact: Element) -> list[Element]:
        data = CacheManager.get_cache().get(configuration=self.__configuration, input_key=artifact.to_json())
        elements: list[Element] = list()
        parent_mapping: dict[str, str] = {}
        for element_dict in data:
            element = Element.element_from_dict(element_dict)
            elements.append(element)
            parent_mapping[element.identifier] = element_dict["parent"]
        for element in elements:
            element.parent = (e for e in elements if e.identifier == parent_mapping[element.identifier])

        return elements

    def preprocess(self, artifact: Artifact) -> list[Element]:
        elements = self.__get_cached(artifact)
        if elements:
            return elements

        # TODO: raise error if content is not text
        element = Element(identifier=artifact.identifier,
                          type=artifact.type,
                          content=str(artifact.content),
                          parent=None,
                          granularity=0)
        elements = [element]

        for element in elements:
            CacheManager.get_cache().put(configuration=self.__configuration, input=artifact.to_json(), data=element.to_dict())

        return elements
