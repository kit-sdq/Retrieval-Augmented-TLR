import json
import os
from base64 import b64encode
from hashlib import shake_128

import dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings

from .embedding_creator import EmbeddingCreator, Element
from ..module import ModuleConfiguration


class OLLAMAEmbeddingCreator(EmbeddingCreator):
    __embedder: Embeddings

    def __init__(self, configuration: ModuleConfiguration):
        store = LocalFileStore(configuration.args.setdefault("path", "./storage/embeddings/"))
        configuration.args.pop("path")
        dotenv.load_dotenv()

        host = os.environ.get("OLLAMA_HOST")
        username = os.environ.get("OLLAMA_USER")
        password = os.environ.get("OLLAMA_PASSWORD")

        if host is None or username is None or password is None:
            raise ValueError("OLLAMA_USER and OLLAMA_PASSWORD must be set in .env file")
        headers = {'Authorization': "Basic " + b64encode(f"{username}:{password}".encode('utf-8')).decode("ascii")}

        embedding_model = OllamaEmbeddings(base_url=host,
                                           model=configuration.args.setdefault("model", "nomic-embed-text:v1.5"),
                                           headers=headers)
        hash = shake_128(configuration.name.encode())
        hash.update(json.dumps(configuration.args, sort_keys=True).encode())
        namespace = hash.hexdigest(31)

        self.__embedder = CacheBackedEmbeddings.from_bytes_store(
            embedding_model, store, namespace=namespace)

    def calculate_embedding(self, element: Element) -> list[float]:
        print("Embedding: " + element.identifier)
        return self.__embedder.embed_documents([element.content])[0]

    def calculate_multiple_embeddings(self, elements: list[Element]) -> list[list[float]]:
        contents = [x.content for x in elements]
        embeddings = [emb for emb in self.__embedder.embed_documents(contents)]
        return embeddings
