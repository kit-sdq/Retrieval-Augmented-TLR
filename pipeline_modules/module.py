import typing
from collections.abc import Mapping


class ModuleConfiguration:
    type: str
    name: str
    args: dict[str, typing.Any]


class PipelineConfiguration:
    source_artifact_provider: ModuleConfiguration
    target_artifact_provider: ModuleConfiguration
    source_preprocessor: ModuleConfiguration
    target_preprocessor: ModuleConfiguration
    embedding_creator: ModuleConfiguration
    source_store: ModuleConfiguration
    target_store: ModuleConfiguration
    classifier: ModuleConfiguration
    result_aggregator: ModuleConfiguration

