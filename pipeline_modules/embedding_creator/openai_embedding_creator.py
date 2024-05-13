import json
import time

import openai

from .embedding_creator import EmbeddingCreator, Element, Embedding
from ..module import ModuleConfiguration

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from hashlib import shake_128


class OpenAIEmbeddingCreator(EmbeddingCreator):
    __embedder: Embeddings

    def __init__(self, configuration: ModuleConfiguration):
        store = LocalFileStore(configuration.args.setdefault("path", "./storage/embeddings/"))
        configuration.args.pop("path")
        embedding_model = OpenAIEmbeddings(model=configuration.args.setdefault("model", "text-embedding-ada-002"))
        hash = shake_128(configuration.name.encode())
        hash.update(json.dumps(configuration.args, sort_keys=True).encode())
        namespace = hash.hexdigest(31)

        self.__embedder = CacheBackedEmbeddings.from_bytes_store(
            embedding_model, store, namespace=namespace)

    def calculate_embedding(self, element: Element) -> Embedding:
        print("Embedding: " + element.identifier)
        try:
            embedding = Embedding(self.__embedder.embed_documents([element.content])[0])
        except openai.RateLimitError as e:
            time.sleep(60)
            embedding = Embedding(self.__embedder.embed_documents([element.content])[0])
        return embedding

    def calculate_multiple_embeddings(self, elements: list[Element]) -> list[Embedding]:
        contents = [x.content for x in elements]
        embeddings = [Embedding(emb) for emb in self.__embedder.embed_documents(contents)]
        return embeddings
