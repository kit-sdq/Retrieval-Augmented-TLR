from hashlib import sha256
from hashlib import shake_128

import chromadb

from .element_store import ElementStore, EmbeddedElement
from ..embedding_creator.embedding_creator import Embedding
from ..knowledge import Element
from ..module import ModuleConfiguration


class ChromaElementStore(ElementStore):
    __N_ALL = "all"
    __N_DYNAMIC = "dynamic"

    __configuration: ModuleConfiguration
    __direction: str
    __n_results: int | str
    __threshold: float
    __use_dynamic_n: bool
    __similarity_function: str
    __compare_length: int = 0
    __db: chromadb.ClientAPI
    __collection: chromadb.Collection | None

    def __init__(self, configuration: ModuleConfiguration):
        self.__configuration = configuration
        self.__direction = self.__configuration.args["direction"]
        self.__n_results = self.__parse_n(self.__configuration.args.get("n_results", 10))
        self.__similarity_function = self.__configuration.args.setdefault("similarity_function", "cosine")
        self.__threshold = self.__configuration.args.setdefault("threshold", 1.0)
        self.__use_dynamic_n = self.__configuration.args.get("dynamic_n", False)

        self.__db = chromadb.PersistentClient(path=self.__configuration.args["path"])

    def __parse_n(self, n: int | str) -> int | str:
        # TODO: is there a nicer way?
        if n == self.__N_ALL:
            return self.__N_ALL
        if n == self.__N_DYNAMIC:
            return self.__N_DYNAMIC
        else:
            return int(n)

    def __config_hash(self) -> str:
        hash = sha256()
        hash.update(self.__configuration.name.encode())
        hash.update(self.__configuration.args["direction"].encode())
        hash.update(self.__configuration.args["similarity_function"].encode())
        hash_hex = hash.hexdigest()
        return hash_hex

    def __collection_name(self, previous_modules_key: str):
        hash = shake_128()
        hash.update(previous_modules_key.encode())
        hash.update(self.__config_hash().encode())

        collection_name = hash.hexdigest(31)
        return collection_name

    def create_vector_store(self,
                            previous_modules_key: str,
                            entries: list[EmbeddedElement]):
        collection_name = self.__collection_name(previous_modules_key)
        self.__collection = self.__db.get_or_create_collection(name=collection_name,
                                                               metadata={"hnsw:space": self.__similarity_function,
                                                                     "hnsw:M": 32,
                                                                     }
                                                               )

        embeddings = list()
        ids = list()
        documents = list()
        metadatas = list()

        # TODO: Check if elements/embeddings are the same. Raise error if not
        if self.__collection.count() == 0:
            for emb_element in entries:
                embeddings.append(emb_element.embedding.embedding)

                element = emb_element.element
                ids.append(element.identifier)
                documents.append(element.content)
                metadata = {"type": element.type,
                            "granularity": element.granularity,
                            "parent": element.parent.identifier if element.granularity != 0 else "",
                            "compare": element.compare}
                metadatas.append(metadata)

            # Split due to maximum batch size of 5461
            start_index = 0
            end_index = min(1000, len(ids))
            while start_index < len(ids):
                self.__collection.add(ids=ids[start_index:end_index],
                                      embeddings=embeddings[start_index:end_index],
                                      documents=documents[start_index:end_index],
                                      metadatas=metadatas[start_index:end_index])
                start_index = end_index
                end_index = min(start_index + 1000, len(ids))

        self.__compare_length = len(self.__collection.get(where={"compare": True})['ids'])

    def find_similar(self, query: Embedding) -> list[Element]:
        elements, distances = self.find_similar_with_distances(query)
        return elements

    def __calculate_dynamic_n(self, distances: list[float]) -> int:
        # TODO: implement or delete
        pass

    def find_similar_with_distances(self, query: Embedding) -> (list[Element], list[float]):
        metadata_filter = {
            "compare": True
        }
        # To ensure determinism: get all elements and cut off later
        results = self.__collection.query(query.embedding, n_results=self.__compare_length, where=metadata_filter,
                                          include=["distances"])
        sorted_results = sorted(zip(results["distances"][0], results["ids"][0]))
        sorted_results = [x for x in sorted_results if x[0] <= self.__threshold]

        if self.__n_results == self.__N_ALL:
            sorted_results = sorted_results
        elif self.__n_results == self.__N_DYNAMIC:
            sorted_results = sorted_results[:self.__calculate_dynamic_n(results["distances"][0])] # TODO: see __calculate_dynamic_n()
        else:
            sorted_results = sorted_results[:self.__n_results]

        elements = list()
        distances = [x[0] for x in sorted_results]
        for result in sorted_results:
            elements.append(self.get_by_id(identifier=result[1]))
        return elements, distances

    requested_elements: dict[str, Element] = {}

    def get_by_id(self, identifier: str) -> Element:
        element = self.requested_elements.get(identifier)
        if element:
            return element

        result = self.__collection.get(ids=identifier)

        identifier = result["ids"][0]
        document = result["documents"][0]
        metadata = result["metadatas"][0]
        element_dict = {
            "identifier": identifier,
            "type": metadata["type"],
            "content": document,
            "granularity": metadata["granularity"],
            "compare": metadata["compare"]
            }
        element = Element.element_from_dict(element=element_dict)
        if element.granularity != 0:
            element.parent = self.get_by_id(metadata["parent"])
        self.requested_elements[identifier] = element
        return element

    def __get_by_metadata_filter(self, metadata_filter: dict[str, str | bool]) -> list[EmbeddedElement]:
        results = self.__collection.get(where=metadata_filter, include=["embeddings"])
        ids = results["ids"]
        embeddings = results["embeddings"]
        elements = list()
        for i in range(len(ids)):
            element = self.get_by_id(ids[i])
            elements.append(EmbeddedElement(element, Embedding(embeddings[i])))
        return elements

    def get_by_parent_id(self, identifier: str) -> list[EmbeddedElement]:
        metadata_filter = {
            "parent": identifier
        }
        return self.__get_by_metadata_filter(metadata_filter)

    def get_all_elements(self, compare: bool = False) -> list[EmbeddedElement]:
        metadata_filter = {}
        if compare:
            metadata_filter["compare"] = True
        return self.__get_by_metadata_filter(metadata_filter)
