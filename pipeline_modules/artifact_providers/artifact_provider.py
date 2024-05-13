from typing import Protocol

from ..knowledge import Artifact
from ..module import ModuleConfiguration


class ArtifactProvider(Protocol):

    def get_all_artifacts(self) -> list[Artifact]:
        ...

    def get_artifact(self, identifier: str) -> Artifact:
        ...


class ArtifactProviderBuilder:
    from .mock_artifact_provider import MockArtifactProvider
    from .text_artifact_provider import TextArtifactProvider
    from .deep_text_artifact_provider import DeepTextArtifactProvider
    from .single_file_artifact_provider import SingleFileArtifactProvider

    ARTIFACT_PROVIDERS = {
        'mock': MockArtifactProvider,
        'text': TextArtifactProvider,
        'deep_text': DeepTextArtifactProvider,
        'single_file': SingleFileArtifactProvider
    }

    def build_artifact_provider(self, configuration: ModuleConfiguration):
        return self.ARTIFACT_PROVIDERS[configuration.name](configuration)
