import tree_sitter

from .preprocessor import Preprocessor
from ..knowledge import Element, Artifact
from ..module import ModuleConfiguration
from project.cache.cache_manager import CacheManager

from tree_sitter import Node, Parser
from tree_sitter_languages import get_language


class CodeMethodPreprocessor(Preprocessor):
    __configuration: ModuleConfiguration
    __language: str

    def __init__(self, configuration: ModuleConfiguration):
        self.__language = configuration.args["language"]
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

    def __get_language(self) -> tree_sitter.Language:
        if self.__language == "java":
            return get_language(self.__language)
        else:
            raise ValueError

    def __get_methods(self, node: Node) -> list[Node]:
        methods = []
        if node.type == 'method_declaration':
            methods.append(node)
        else:
            for child in node.children:
                methods += self.__get_methods(child)
        return methods

    def __get_class_bodies(self, node: Node) -> list[Node]:
        if node.type == 'class_body':
            return [node]
        else:
            classes = list()
            for child in node.children:
                classes += self.__get_class_bodies(child)
            return classes

    def preprocess(self, artifact: Artifact) -> list[Element]:
        elements = self.__get_cached(artifact)
        if elements:
            return elements
        elements.append(artifact)

        parser = Parser()
        parser.set_language(get_language(self.__language))

        tree = parser.parse(artifact.content.encode())

        i = 0
        class_start = 0
        for class_body in self.__get_class_bodies(tree.root_node):
            text = str(tree.text[class_start:class_body.start_byte])
            class_element = Element(identifier=artifact.identifier + "$" + str(i),
                                    type="source code class definition",
                                    content=text,
                                    parent=artifact,
                                    granularity=1,
                                    compare=False)
            elements.append(class_element)

            method_start = class_body.start_byte
            j = 0
            for method in self.__get_methods(class_body):
                method_text = str(tree.text[method_start:method.end_byte])
                method_element = Element(identifier=class_element.identifier + "$" + str(j),
                                         type="source code method",
                                         content=method_text,
                                         parent=class_element,
                                         granularity=class_element.granularity + 1,
                                         compare=True)
                elements.append(method_element)
                method_start = method.end_byte
                j = j + 1

            class_start = class_body.end_byte
            i = i + 1

        for element in elements:
            CacheManager.get_cache().put(configuration=self.__configuration, input=artifact.to_json(),
                                         data=element.to_dict())
        return elements
