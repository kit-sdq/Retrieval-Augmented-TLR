import json
import os
from base64 import b64encode

from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from cache.cache_manager import CacheManager
from .classifier import Classifier, ClassificationResult, Element
from .context_provider import ContextProvider
from ..module import ModuleConfiguration


class SimpleOllamaClassifier(Classifier):
    """A classifier comparing a single source and a single target element at a time, not using additional context."""
    __configuration: ModuleConfiguration
    __prompt: ChatPromptTemplate
    __llm: ChatOllama
    __parser: StrOutputParser
    __chain: Runnable

    __context_provider: ContextProvider

    def __init__(self, configuration: ModuleConfiguration, context_provider: ContextProvider):
        self.__configuration = configuration
        self.__context_provider = context_provider
        self.__setup_prompt()

        host = os.environ.get("OLLAMA_HOST")
        username = os.environ.get("OLLAMA_USER")
        password = os.environ.get("OLLAMA_PASSWORD")

        if host is None or username is None or password is None:
            raise ValueError("OLLAMA_USER and OLLAMA_PASSWORD must be set in .env file")

        headers = {'Authorization': "Basic " + b64encode(f"{username}:{password}".encode('utf-8')).decode("ascii")}

        self.__llm = ChatOllama(base_url=host,model=self.__configuration.args.setdefault("model", "llama3"), temperature=0,
                                headers=headers)
        self.__parser = StrOutputParser()
        self.__chain = self.__prompt | self.__llm | self.__parser

    def __setup_prompt(self):
        template = """Question: Here are two parts of software development artifacts. \n
        {source_type}: '''{source_content}''' \n
        {target_type}: '''{target_content}'''
        Are they related? \n
        Answer with 'yes' or 'no'.
        """

        self.__prompt = ChatPromptTemplate.from_template(template)

    def __get_input_key(self, source: Element, target: Element) -> str:
        inputs = {
            "source": source.to_dict(),
            "target": target.to_dict()
        }
        input_key = json.dumps(inputs, sort_keys=True)
        return input_key

    def __get_cached_related(self, source: Element, target: Element) -> bool | None:
        input_key = self.__get_input_key(source, target)
        data = CacheManager.get_cache().get(configuration=self.__configuration, input_key=input_key)
        if data:
            related = data[0]["related"]
            return related

        return None

    def __cache_target(self, source: Element, target: Element, output: str, related: bool):
        data = {
            "source": source.identifier,
            "target": target.identifier,
            "output": output,
            "related": related
        }
        CacheManager.get_cache().put(configuration=self.__configuration, input=self.__get_input_key(source, target),
                                     data=data)

    def __is_related(self, output: str) -> bool:
        related = "yes" in output.lower()
        return related

    def classify(self, source: Element, targets: list[Element]) -> ClassificationResult:
        related_targets = list()

        inputs = list()
        for target in targets:
            related = self.__get_cached_related(source, target)
            if related is not None:
                if related:
                    related_targets.append(target)
            else:
                print("Invoking the LLM for " + source.identifier + " : " + target.identifier)
                inputs.append((target,
                               {"source_type": source.type,
                                "target_type": target.type,
                                "source_content": source.content,
                                "target_content": target.content}))

        outputs = self.__chain.batch(inputs=[x[1] for x in inputs])
        results = zip([x[0] for x in inputs], outputs)
        for result in results:
            related = self.__is_related(result[1])
            if related:
                related_targets.append(result[0])
            self.__cache_target(source=source, target=result[0], output=result[1], related=related)

        return ClassificationResult(source, related_targets)
