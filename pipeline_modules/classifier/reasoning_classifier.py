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


class ReasoningClassifier(Classifier):
    """A classifier comparing a single source and a single target element at a time,
    prompting the LLM to reason about it's decision"""
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
        self.context_provider = context_provider
        self.__system__message = configuration.args.setdefault("system_message", True)
        self.__use_original_artifacts = configuration.args.setdefault("use_original_artifacts", False)

        self.prompt_number = configuration.args.get("prompt_number", 0)

        self.__setup_prompt()
        self.llm = ChatOpenAI(model=configuration.args.setdefault("model", "gpt-3.5-turbo-0125"), temperature=0, max_tokens=1024)
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser
        self.__configuration = configuration

    def __setup_prompt(self):
        """The prompt is based on the prompts presented by Rodriguez et al. in
        Prompts Matter: Insights and Strategies for Prompt Engineering in Automated Software Traceability (2023)"""

        # TODO: refactor, similar to MultiStep
        system_message = ("system",
                          """Your job is to determine if there is a traceability link between two artifacts of a system.""")
        message = ("user",
                   """Below are two artifacts from the same software system. Is there a traceability link between (1) and (2)? Give your reasoning and then answer with 'yes' or 'no' enclosed in <trace> </trace>.\n (1) {source_type}: '''{source_content}''' \n (2) {target_type}: '''{target_content}''' """)

        if self.prompt_number == 1:
            message = ("user",
                       """Below are two artifacts from the same software system. Is there a conceivable traceability link between (1) and (2)? Give your reasoning and then answer with 'yes' or 'no' enclosed in <trace> </trace>.\n (1) {source_type}: '''{source_content}''' \n (2) {target_type}: '''{target_content}''' """)
        if self.prompt_number == 2:
            message = ("user",
                       """Below are two artifacts from the same software system.\n Give one reason why (1) might be related to (2) enclosed in <related> </related>.\n Give one reason why (1) might not be related to (2) enclosed in <unrelated> </unrelated>.\n Then answer: Is there a conceivable traceability link between (1) and (2)? Answer with 'yes' or 'no' enclosed in <trace> </trace>.\n (1) {source_type}: '''{source_content}''' \n (2) {target_type}: '''{target_content}''' """)
        if self.prompt_number == 3:
            message = ("user",
                       """Below are two artifacts from the same software system.\n Give one reason why (1) might not be related to (2) enclosed in <unrelated> </unrelated>.\n Give one reason why (1) might be related to (2) enclosed in <related> </related>.\n Then answer: Is there a conceivable traceability link between (1) and (2)? Answer with 'yes' or 'no' enclosed in <trace> </trace>.\n (1) {source_type}: '''{source_content}''' \n (2) {target_type}: '''{target_content}''' """)
        if self.prompt_number == 4:
            message = ("user",
                       """Below are two artifacts from the same software system.\n Is there a traceability link between (1) and (2)? Give your reasoning and then answer with 'yes' or 'no' enclosed in <trace> </trace>. Only answer yes if you are absolutely certain.\n (1) {source_type}: '''{source_content}''' \n (2) {target_type}: '''{target_content}''' """)

        messages = list()
        if self.__system__message:
            messages.append(system_message)
        messages.append(message)
        self.prompt = ChatPromptTemplate.from_messages(messages)

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
        match = re.search("<trace>(.*?)</trace>", output.lower())
        related = False
        if match:
            related = "yes" in match.group()
        return related

    def classify(self, source: Element, targets: list[Element]) -> ClassificationResult:
        related_targets = list()

        inputs = list()
        invoke_source = source
        invoke_targets = list()
        if self.__use_original_artifacts:
            invoke_source = source.original_artifact()
            for target in targets:
                original = target.original_artifact()
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
            related = self.__is_related(result[1])
            if related:
                related_targets.append(result[0])
            self.__cache_target(source=invoke_source, target=result[0], output=result[1], related=related)

        return ClassificationResult(invoke_source, related_targets)
