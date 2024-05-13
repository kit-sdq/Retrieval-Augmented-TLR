from .preprocessor import Preprocessor
from ..knowledge import Element, Artifact
from ..module import ModuleConfiguration
from project.cache.cache_manager import CacheManager
import pysbd


class SentenceSplitterPreprocessor(Preprocessor):
    """Split artifact text into Elements containing a single sentence."""
    __configuration: ModuleConfiguration
    __language: str
    __clean: bool

    def __init__(self, configuration: ModuleConfiguration):
        self.__language = configuration.args.setdefault("language", "en")
        self.__clean = configuration.args.setdefault("clean_text", False)
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

        segmenter = pysbd.Segmenter(language=self.__language, clean=self.__clean)
        segments: list[str] = segmenter.segment(artifact.content)

        i = 0
        for segment in segments:
            element = Element(identifier=artifact.identifier + "$" + str(i),
                              type=artifact.type,
                              content=segment,
                              parent=artifact,
                              granularity=1,
                              compare=True)
            elements.append(element)
            i = i + 1

        for element in elements:
            CacheManager.get_cache().put(configuration=self.__configuration, input=artifact.to_json(),
                                         data=element.to_dict())
        return elements
