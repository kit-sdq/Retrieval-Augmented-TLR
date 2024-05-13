from .preprocessor import Preprocessor
from ..knowledge import Element, Artifact
from ..module import ModuleConfiguration
from project.cache.cache_manager import CacheManager
import pysbd


class LineSplitterPreprocessor(Preprocessor):
    """A preprocessor splitting an artifact into elements containing a single line."""
    __configuration: ModuleConfiguration

    def __init__(self, configuration: ModuleConfiguration):
        self.__configuration = configuration
        pass

    def __get_cached(self, artifact: Element) -> list[Element]:
        # TODO: refactor out caching to another class (same as simple_text_preprocessor)

        data = CacheManager.get_cache().get(configuration=self.__configuration, input_key=artifact.to_json())
        elements: list[Element] = list()
        parent_mapping: dict[str, str] = {}

        for element_dict in data:
            element = Element.element_from_dict(element_dict)
            elements.append(element)
            parent_mapping[element.identifier] = element_dict["parent"]

        for element in elements:
            if parent_mapping[element.identifier] is not None:
                element.parent = [e for e in elements if e.identifier == parent_mapping[element.identifier]][0]
            else:
                element.parent = None
        return elements

    def preprocess(self, artifact: Artifact) -> list[Element]:
        elements = self.__get_cached(artifact)
        if elements:
            return elements
        elements.append(artifact)

        segments = artifact.content.splitlines()
        segments = [segment for segment in segments if segment != ""]
        for i in range(1, len(segments)+1):
            element = Element(identifier=artifact.identifier + "$" + str(i),
                              type=artifact.type,
                              content=segments[i-1],
                              parent=artifact,
                              granularity=1,
                              compare=True)
            elements.append(element)

        for element in elements:
            CacheManager.get_cache().put(configuration=self.__configuration, input=artifact.to_json(),
                                         data=element.to_dict())
        return elements
