import itertools
import json
from hashlib import sha256

from pipeline_modules.module import PipelineConfiguration, ModuleConfiguration

from pipeline_modules.artifact_providers.artifact_provider import ArtifactProvider, ArtifactProviderBuilder
from pipeline_modules.classifier.context_provider import ContextProvider
from pipeline_modules.classifier.classifier import Classifier, ClassifierBuilder
from pipeline_modules.element_store.element_store import ElementStore, ElementStoreBuilder, EmbeddedElement
from pipeline_modules.embedding_creator.embedding_creator import EmbeddingCreator, EmbeddingCreatorBuilder, Embedding
from pipeline_modules.preprocessors.preprocessor import Preprocessor, PreprocessorBuilder
from pipeline_modules.result_aggregator.result_aggregator import ResultAggregator, ResultAggregatorBuilder
from pipeline_modules.result_aggregator.result_aggregator import TraceLink


class Controller:
    source_preprocessor_config: ModuleConfiguration
    target_preprocessor_config: ModuleConfiguration
    embedding_config: ModuleConfiguration

    source_artifact_provider: ArtifactProvider
    target_artifact_provider: ArtifactProvider
    source_preprocessor: Preprocessor
    target_preprocessor: Preprocessor
    embedding_creator: EmbeddingCreator
    source_store: ElementStore
    target_store: ElementStore
    context_provider: ContextProvider
    classifier: Classifier
    result_aggregator: ResultAggregator

    def __init__(self, pipeline_configuration: PipelineConfiguration):
        # Special handling for preprocessors and embedding to provide hash for later modules.
        self.source_preprocessor_config = pipeline_configuration.source_preprocessor
        self.target_preprocessor_config = pipeline_configuration.target_preprocessor
        self.embedding_config = pipeline_configuration.embedding_creator

        self.source_artifact_provider = ArtifactProviderBuilder().build_artifact_provider(
            configuration=pipeline_configuration.source_artifact_provider)
        self.target_artifact_provider = ArtifactProviderBuilder().build_artifact_provider(
            configuration=pipeline_configuration.target_artifact_provider)

        self.source_preprocessor = PreprocessorBuilder().build_preprocessor(
            configuration=pipeline_configuration.source_preprocessor)
        self.target_preprocessor = PreprocessorBuilder().build_preprocessor(
            configuration=pipeline_configuration.target_preprocessor)

        self.embedding_creator = EmbeddingCreatorBuilder().build_embedding_creator(
            configuration=pipeline_configuration.embedding_creator)

        self.source_store = ElementStoreBuilder().build_element_store(
            configuration=pipeline_configuration.source_store)
        self.target_store = ElementStoreBuilder().build_element_store(
            configuration=pipeline_configuration.target_store)

        self.context_provider = ContextProvider(self.source_store, self.target_store)
        self.classifier = ClassifierBuilder().build_classifier(
            configuration=pipeline_configuration.classifier, context_provider=self.context_provider)

        self.result_aggregator = ResultAggregatorBuilder().build_result_aggregator(
            configuration=pipeline_configuration.result_aggregator)

    def __preprocessor_embedding_key(self, source: bool) -> str:
        hash = sha256()
        hash.update((self.source_preprocessor_config.name if source else self.target_preprocessor_config.name).encode())
        hash.update(json.dumps(self.source_preprocessor_config.args if source else self.target_preprocessor_config.args,
                               sort_keys=True).encode())
        return hash.hexdigest()

    def run(self) -> list[TraceLink]:  # TODO: refactor into smaller functions
        print("Controller running...")

        # Preprocess target artifacts
        print("Target Artifact Provider")
        target_artifacts = self.target_artifact_provider.get_all_artifacts()
        print("Target Preprocessor")
        target_elements = list(itertools.chain.from_iterable(map(self.target_preprocessor.preprocess, target_artifacts)))

        print("Target Embedding Creator")
        target_embeddings: list[EmbeddedElement] = [
            EmbeddedElement(element=element, embedding=embedding)
            for element, embedding
            in zip(target_elements, self.embedding_creator.calculate_multiple_embeddings(elements=target_elements))
        ]

        print("Target Element Store")
        self.target_store.create_vector_store(previous_modules_key=self.__preprocessor_embedding_key(False),
                                              entries=target_embeddings)

        # Preprocess source artifacts  # TODO: same as target artifacts -> refactor
        print("Source Artifact Provider")
        source_artifacts = self.source_artifact_provider.get_all_artifacts()
        print("Source Preprocessor")
        source_elements = list(
            itertools.chain.from_iterable(map(self.source_preprocessor.preprocess, source_artifacts)))
        print("Source Embedding Creator")
        source_embeddings: list[EmbeddedElement] = [
            EmbeddedElement(element=element, embedding=embedding)
            for element, embedding
            in zip(source_elements, self.embedding_creator.calculate_multiple_embeddings(elements=source_elements))
        ]
        print("Source Element Store")
        self.source_store.create_vector_store(previous_modules_key=self.__preprocessor_embedding_key(True),
                                              entries=source_embeddings)

        # Classification
        print("Classifier")
        classification_results = []
        for query in self.source_store.get_all_elements(compare=True):
            target_candidates = self.target_store.find_similar(query=query.embedding)
            print(f"{query.element.identifier} : {target_candidates}")

            classification_results.append(self.classifier.classify(query.element, target_candidates))

        print("Result Aggregator")
        trace_links = self.result_aggregator.aggregate(classification_results)

        return list(trace_links)
