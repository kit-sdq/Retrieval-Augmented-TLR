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
        self._configuration = configuration
        self._path = self._configuration.args["path"]
        self._direction = self._configuration.args["direction"]

        if self._path is None or self._direction is None:
            raise ValueError("Path and direction have to be set in the configuration")

        self._max_results = self._configuration.args.get("max_results", 10)
        self._embedded_elements: dict[str, EmbeddedElement] = {}

    def _create_id(self, previous_modules_key: str) -> str:
        module_id = shake_128()
        module_id.update(self._configuration.name.encode())
        module_id.update(self._direction.encode())
        module_id.update(previous_modules_key.encode())
        module_id = module_id.hexdigest(31)
        return module_id

    def create_vector_store(self,
                            previous_modules_key: str,
                            entries: list[EmbeddedElement]):
        path = self._path + "/ces/" + self._create_id(previous_modules_key)
        create_new = not os.path.exists(path) or len(os.listdir(path)) == 0
        if create_new:
            print("Creating new elements")
            os.makedirs(path, exist_ok=True)
            self._embedded_elements = {entry.element.identifier: entry for entry in entries}

            if len(self._embedded_elements) != len(entries):
                raise ValueError("Duplicate identifiers in entries")

            # Save Elements
            for i, entry in enumerate(entries):
                with open(os.path.join(path, f"{i}.json"), 'w') as f:
                    json_element = entry.element.to_dict()
                    json_element["embedding"] = entry.embedding
                    f.write(dumps(json_element))
        else:
            # TODO: Check if elements/embeddings are the same. Raise error if not
            self._embedded_elements = self.__load_elements(path)

    def __load_elements(self, path) -> dict[str, EmbeddedElement]:
        print("Loading existing elements")
        elements: dict[str, EmbeddedElement] = {}
        for root, _, files in os.walk(path):
            for file in files:
                with open(os.path.join(root, file), 'r') as f:
                    json_element = loads(f.read())
                    element = Element.element_from_dict(json_element)
                    embedding: list[float] = json_element["embedding"]
                    elements[element.identifier] = EmbeddedElement(element, embedding)
                    element.parent = json_element["parent"]  # Wrong type (only id) .. will be fixed below

        for embedded_element in elements.values():
            parent_id = embedded_element.element.parent  # is string because loaded from json
            if parent_id is not None:
                embedded_element.element.parent = elements[parent_id].element

        return elements

    def find_similar(self, query: list[float]) -> list[Element]:
        elements, _ = self.find_similar_with_distances(query)
        return elements

    def find_similar_with_distances(self, query: list[float]) -> (list[Element], list[float]):
        # Cosine Similarity
        results = self.__cosine_similarity(query)
        return results

    def __cosine_similarity(self, query: list[float]) -> (list[Element], list[float]):
        results = []
        for element in self.get_all_elements(compare=True):
            similarity = dot(query, element.embedding) / (norm(query) * norm(element.embedding))
            results.append((element.element, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return [result[0] for result in results[:self._max_results]], [result[1] for result in
                                                                       results[:self._max_results]]

    def get_by_id(self, identifier: str) -> Element:
        return self._embedded_elements[identifier].element

    def get_by_parent_id(self, identifier: str) -> list[EmbeddedElement]:
        elements = [e for e in self._embedded_elements.values() if
                    e.element.parent and e.element.parent.identifier == identifier]
        return list(sorted(elements, key=lambda x: x.element.identifier))

    def get_all_elements(self, compare: bool = False) -> list[EmbeddedElement]:
        # If true, only return elements that are marked for comparison otherwise all elements
        elements = [element for element in self._embedded_elements.values() if element.element.compare or not compare]
        return list(sorted(elements, key=lambda x: x.element.identifier))
