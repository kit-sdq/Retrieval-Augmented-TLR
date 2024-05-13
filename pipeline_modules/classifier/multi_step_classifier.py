import json
import re
from enum import Enum
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .classifier import Classifier, ClassificationResult, Element
from .context_provider import ContextProvider
from ..module import ModuleConfiguration

from project.cache.cache_manager import CacheManager


class StepResult(Enum):
    """Result status used after a prompting step. UNRELATED and RELATED are used for early stopping."""
    UNRELATED = 0
    RELATED = 1
    CONTINUE = 2


class PromptStep:
    """Used for MultiStepClassifier without branching prompts."""
    messages: list[(str, str)]
    status: Callable[[str], StepResult]

    def __init__(self, message_templates: list[(str, str)], status: Callable[[str], StepResult]):
        self.messages = message_templates
        self.status = status

    def template_json(self) -> str:
        return json.dumps(self.messages, sort_keys=True)


class DefaultMultiStepPrompts:
    @staticmethod
    def callable_contains_tag(tag: str, text: str, positive: StepResult, negative: StepResult) -> Callable[[str], StepResult]:
        """Checks whether a <tag>value</tag> with a given tag exists
        and whether the given text exists within the encased value."""
        def contains_string(output: str) -> StepResult:
            match = re.search(f"<{tag}>(.*?)</{tag}>", output.lower())
            if match:
                if text.lower() in match.group(1):
                    return positive
                else:
                    return negative
            return negative

        return contains_string

    prompts: dict[str, list[PromptStep]] = {
        "is_component_reasoning": [
            PromptStep(
                [
                    ("system", "Your job is to determine if an artifact of a software system describes a component."),
                    ("user", "You are given a part of a {source_type}. Does it refer specifically to a component? Give your reasoning and then answer with '<component>yes</component>' or '<component>no</component>'. \n {source_type}: \n'''{source_content}'''")
                ],
                callable_contains_tag("component", "yes", StepResult.CONTINUE, StepResult.UNRELATED)
            ),
            PromptStep(
                [
                    ("system", "Your job is to determine if there is a traceability link between two artifacts of a system."),
                    ("user", """Below are two artifacts from the same software system. Is there a traceability link between (1) and (2)? Give your reasoning and then answer with 'yes' or 'no' enclosed in <trace> </trace>.\n (1) {source_type}: '''{source_content}''' \n (2) {target_type}: '''{target_content}''' """)
                ],
                callable_contains_tag("trace", "yes", StepResult.RELATED, StepResult.UNRELATED)
            )
        ],
        "source_neighbouring_siblings_reasoning": [
            PromptStep(
                [
                    ("system", "Your job is to determine if there is a traceability link between two artifacts of a system."),
                    ("user", """Below are two artifacts from the same software system. Is there a traceability link between (1) and (2)? Give your reasoning and then answer with 'yes' or 'no' enclosed in <trace> </trace>.\n (1) {source_type}: '''{source_content}''' \n (2) {target_type}: '''{target_content}''' \n\n (1) is surrounded by this:\n {source_context_pre}\n{source_content}\n{source_context_post}""")
                ],
                callable_contains_tag("trace", "yes", StepResult.RELATED, StepResult.UNRELATED)
            )
        ]
    }

    @staticmethod
    def get_prompts(identifier: str) -> list[PromptStep]:
        return DefaultMultiStepPrompts.prompts[identifier]


class MultiStepClassifier(Classifier):
    """A classifier comparing a single source and a single target element at a time,
    prompting the LLM multiple times, deciding whether to continue after every LLM call.
    It can use sibling elements to provide context."""

    __configuration: ModuleConfiguration
    __prompts: list[PromptStep]
    __langchain_prompts: list[ChatPromptTemplate]
    __llm: BaseChatModel
    __parser: StrOutputParser
    __chains: list[Runnable]

    __use_original_artifacts: bool

    __context_provider: ContextProvider

    def __init__(self, configuration: ModuleConfiguration, context_provider: ContextProvider):
        self.__context_provider = context_provider
        self.__use_original_artifacts = configuration.args.setdefault("use_original_artifacts", False)

        self.__source_pre_context = configuration.args.get("source_pre_context", 0)
        self.__source_post_context = configuration.args.get("source_post_context", 0)
        self.__target_pre_context = configuration.args.get("target_pre_context", 0)
        self.__target_post_context = configuration.args.get("target_post_context", 0)

        self.__prompts = DefaultMultiStepPrompts.get_prompts(configuration.args.get("prompt"))
        self.__setup_prompts()

        self.__llm = ChatOpenAI(model=configuration.args.setdefault("model", "gpt-3.5-turbo-0125"), temperature=0,
                                max_tokens=1024)
        self.__parser = StrOutputParser()

        self.__chains = list()
        for prompt in self.__langchain_prompts:
            self.__chains.append(prompt | self.__llm | self.__parser)
        self.__configuration = configuration

    def __setup_prompts(self):
        self.__langchain_prompts = list()

        for prompt in self.__prompts:
            self.__langchain_prompts.append(ChatPromptTemplate.from_messages(prompt.messages))

    def __fill_template(self, prompt_template: str, input: dict[str, str]) -> str:
        filled = prompt_template.format_map(input)
        return filled

    def __get_input_key(self, prompt_template: str, input: dict[str, str]) -> str:
        inputs = {
            "prompt": self.__fill_template(prompt_template, input)
        }
        input_key = json.dumps(inputs, sort_keys=True)
        return input_key

    def __get_config_key(self) -> ModuleConfiguration:
        # TODO: change to calculate once and reuse -> small performance increase
        config = ModuleConfiguration()
        config.name = self.__configuration.name
        config.args["use_original_artifacts"] = self.__use_original_artifacts
        config.args["model"] = self.__configuration.args["model"]
        return config

    def __get_cached(self, prompt_template: str, input: dict[str, str]) -> dict | None:
        input_key = self.__get_input_key(prompt_template, input)
        data = CacheManager.get_cache().get(configuration=self.__configuration, input_key=input_key)
        if data:
            return data[0]
        return None

    def __cache_target(self, prompt_template: str, input: dict[str, str], source: Element, target: Element, output: str):
        data = {
            "source": source.identifier,
            "target": target.identifier,
            "output": output,
        }
        CacheManager.get_cache().put(configuration=self.__configuration,
                                     input=self.__get_input_key(prompt_template, input),
                                     data=data)

    def original_artifact(self, element: Element) -> Element:
        while element.granularity > 0:
            element = element.parent
        return element

    def __get_relevant_neighbouring_sibling_context(self, element: Element, pre: int, post: int, is_source: bool, prompt: PromptStep) \
            -> (str, str):
        if pre == 0 and post == 0:
            return "", ""

        prefix = "source" if is_source else "target"
        pre_used = False
        post_used = False
        for message in prompt.messages:
            pre_used = pre_used or f"{{{prefix}_context_pre}}" in message[1]
            post_used = post_used or f"{{{prefix}_context_post}}" in message[1]

        relevant_pre = pre if pre_used else 0
        relevant_post = post if post_used else 0
        return self.__context_provider.neighbouring_sibling_context(is_source, element, relevant_pre, relevant_post)

    def __step(self, index: int, source: Element, targets: list[Element]) -> (list[Element], list[Element]):
        related_targets = list()
        continue_targets = list()

        source_pre, source_post = self.__get_relevant_neighbouring_sibling_context(element=source,
                                                                                   pre=self.__source_pre_context,
                                                                                   post=self.__source_post_context,
                                                                                   is_source=True,
                                                                                   prompt=self.__prompts[index])
        for target in targets:
            target_pre, target_post = self.__get_relevant_neighbouring_sibling_context(element=target,
                                                                                       pre=self.__target_pre_context,
                                                                                       post=self.__target_post_context,
                                                                                       is_source=False,
                                                                                       prompt=self.__prompts[index])
            input = {
                "source_type": source.type,
                "target_type": target.type,
                "source_content": source.content,
                "target_content": target.content,
                "source_context_pre": source_pre,
                "source_context_post": source_post,
                "target_context_pre": target_pre,
                "target_context_post": target_post
            }
            data = self.__get_cached(self.__prompts[index].template_json(), input)
            if data is not None:
                status = self.__prompts[index].status(data['output'])
                if status == StepResult.RELATED:
                    related_targets.append(target)
                elif status == StepResult.CONTINUE:
                    continue_targets.append(target)
            else:
                print("Invoking the LLM for " + source.identifier + " : " + target.identifier)
                output = self.__chains[index].invoke(input)
                self.__cache_target(prompt_template=self.__prompts[index].template_json(),
                                    input=input,
                                    source=source,
                                    target=target,
                                    output=output)
                status = self.__prompts[index].status(output)
                if status == StepResult.RELATED:
                    related_targets.append(target)
                elif status == StepResult.CONTINUE:
                    continue_targets.append(target)

        return related_targets, continue_targets

    def classify(self, source: Element, targets: list[Element]) -> ClassificationResult:
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

        related = list()
        for i in range(len(self.__prompts)):
            related_targets, continue_targets = self.__step(index=i, source=invoke_source, targets=invoke_targets)
            related += related_targets
            invoke_targets = continue_targets

        return ClassificationResult(invoke_source, related)
