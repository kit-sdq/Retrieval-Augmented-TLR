from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

from cache.cache_manager import CacheManager
from .preprocessor import Preprocessor
from ..knowledge import Element, Artifact
from ..module import ModuleConfiguration


class CodeChunkingPreprocessor(Preprocessor):
    __configuration: ModuleConfiguration
    __language: str
    __chunk_size: int

    def __init__(self, configuration: ModuleConfiguration):
        self.__language = configuration.args["language"]
        self.__chunk_size = configuration.args.setdefault("chunk_size", 60)
        self.__configuration = configuration
        pass

    def __get_cached(self, artifact: Element) -> list[Element]:
        # TODO: refactor out caching to another class (same as simple_text_preprocessor and sentence_preprocessor)

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

    def __get_language(self) -> Language:
        if self.__language == "java":
            return Language.JAVA
        else:
            raise ValueError

    def preprocess(self, artifact: Artifact) -> list[Element]:
        elements = self.__get_cached(artifact)
        if elements:
            return elements
        elements.append(artifact)

        splitter = RecursiveCharacterTextSplitter.from_language(language=self.__get_language(),
                                                                chunk_size=self.__chunk_size,
                                                                chunk_overlap=0)
        segments = splitter.split_text(artifact.content)

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
