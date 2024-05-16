from .element_store import ElementStore, EmbeddedElement
from ..knowledge import Element
from ..module import ModuleConfiguration
import os
from json import loads, dumps
from numpy import dot
from numpy.linalg import norm
from hashlib import shake_128


class CustomElementStore(ElementStore):

    def __init__(self, configuration: ModuleConfiguration):
        self.__configuration = configuration
        self.__path = self.__configuration.args["path"]
        self.__max_results = self.__configuration.args.get("max_results", 10)
        self.__similarity_function = self.__configuration.args.get("similarity_function", "cosine")
        self.__elements: dict[str, EmbeddedElement] = {}

    def __config_hash(self, previous_modules_key: str) -> str:
        hash = shake_128()
        hash.update(self.__configuration.name.encode())
        hash.update(self.__configuration.args["direction"].encode())
        hash.update(self.__similarity_function.encode())
        hash.update(previous_modules_key.encode())
        hash = hash.hexdigest(31)
        return hash

    def __load_elements(self, path) -> dict[str, EmbeddedElement]:
        elements: dict[str, EmbeddedElement] = {}
        for root, _, files in os.walk(path):
            for file in files:
                with open(os.path.join(root, file), 'r') as f:
                    json_element = loads(f.read())
                    element = Element.element_from_dict(json_element)
                    embedding: list[float] = json_element["embedding"]
                    elements[element.identifier] = EmbeddedElement(element, embedding)
                    element.parent = json_element["parent"]  # Wrong type .. has to be fixed
        for embedded_element in elements.values():
            parent_id = embedded_element.element.parent  # is string
            if parent_id is not None:
                embedded_element.element.parent = elements[parent_id].element
        return elements

    def create_vector_store(self,
                            previous_modules_key: str,
                            entries: list[EmbeddedElement]):
        path = self.__path + "/" + self.__config_hash(previous_modules_key)
        create_new = not os.path.exists(path) or len(os.listdir(path)) == 0
        if create_new:
            print("Creating new elements")
            os.makedirs(path, exist_ok=True)
            self.__elements = {entry.element.identifier: entry for entry in entries}
            for i, entry in enumerate(entries):
                with open(os.path.join(path, f"{i}.json"), 'w') as f:
                    json_element = entry.element.to_dict()
                    json_element["embedding"] = entry.embedding
                    f.write(dumps(json_element))
        else:
            # TODO: Check if elements/embeddings are the same. Raise error if not
            print("Loading existing elements")
            self.__elements = self.__load_elements(path)

    def find_similar(self, query: list[float]) -> list[Element]:
        elements, _ = self.find_similar_with_distances(query)
        return elements

    def find_similar_with_distances(self, query: list[float]) -> (list[Element], list[float]):
        if self.__similarity_function == "cosine":
            results = self.__cosine_similarity(query)
            return results
        else:
            raise NotImplementedError

    def __cosine_similarity(self, query: list[float]) -> (list[Element], list[float]):
        results = []
        for element in self.__elements.values():
            similarity = dot(query, element.embedding) / (norm(query) * norm(element.embedding))
            results.append((element.element, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return [result[0] for result in results[:self.__max_results]], [result[1] for result in
                                                                        results[:self.__max_results]]

    def get_by_id(self, identifier: str) -> Element:
        return self.__elements[identifier].element

    def get_by_parent_id(self, identifier: str) -> list[EmbeddedElement]:
        return [element for element in self.__elements.values() if element.element.parent.identifier == identifier]

    def get_all_elements(self, compare: bool = False) -> list[EmbeddedElement]:
        return [element for element in self.__elements.values() if element.element.compare == compare]
