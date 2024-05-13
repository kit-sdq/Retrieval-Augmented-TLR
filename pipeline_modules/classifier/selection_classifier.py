import json
import re

from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from cache.cache_manager import CacheManager
from .classifier import Classifier, ClassificationResult, Element
from .context_provider import ContextProvider
from ..module import ModuleConfiguration


class SelectionClassifier(Classifier):
    """A classifier comparing a single source and multiple target elements at a time,
    prompting the LLM to select corresponding target elements."""
    # TODO: IMPLEMENT THIS CLASS
    __configuration: ModuleConfiguration
    prompt: ChatPromptTemplate
    llm: ChatOpenAI
    parser: StrOutputParser
    chain: Runnable
    prompt_number: int

    __system__message: bool
    __use_original_artifacts: bool

    context_provider: ContextProvider

    def __init__(self, configuration: ModuleConfiguration, context_provider: ContextProvider):
        pass

    def __setup_prompt(self):
        pass

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

    def is_related(self, output: str) -> bool:
        match = re.search("<trace>(.*?)</trace>", output.lower())
        related = False
        if match:
            related = "yes" in match.group()
        return related

    def original_artifact(self, element: Element) -> Element:
        while element.granularity > 0:
            element = element.parent
        return element

    def classify(self, source: Element, targets: list[Element]) -> ClassificationResult:
        related_targets = list()

        inputs = list()
        invoke_source = source
        invoke_targets = list()
        if self.__use_original_artifacts:
            invoke_source = self.original_artifact(source)
            for target in targets:
                original = self.original_artifact(target)
                if original.identifier not in [element.identifier for element in invoke_targets]:
                    invoke_targets.append(original)
        else:
            invoke_targets = targets

        for target in invoke_targets:
            related = self.__get_cached_related(invoke_source, target)
            if related is not None:
                if related:
                    related_targets.append(target)
            else:
                print("Invoking the LLM for " + invoke_source.identifier + " : " + target.identifier)
                inputs.append((target,
                               {"source_type": invoke_source.type,
                                "target_type": target.type,
                                "source_content": invoke_source.content,
                                "target_content": target.content}))

        outputs = self.chain.batch(inputs=[x[1] for x in inputs])
        results = zip([x[0] for x in inputs], outputs)
        for result in results:
            related = self.is_related(result[1])
            if related:
                related_targets.append(result[0])
            self.__cache_target(source=invoke_source, target=result[0], output=result[1], related=related)

        return ClassificationResult(invoke_source, related_targets)
