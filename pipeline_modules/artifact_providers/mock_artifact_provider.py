from .artifact_provider import ArtifactProvider
from ..knowledge import Artifact
from ..module import ModuleConfiguration

class MockArtifactProvider(ArtifactProvider):
    artifacts = [
        Artifact(identifier="0", type="requirement", content="Lorem ipsum dolor sit amet"),
        Artifact(identifier="1", type="requirement", content="consetetur sadipscing elitr"),
        Artifact(identifier="2", type="requirement", content="sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat")
    ]

    def __init__(self, configuration: ModuleConfiguration):
        pass

    def get_all_artifacts(self) -> list[Artifact]:
        return self.artifacts

    def get_artifact(self, identifier: str) -> Artifact:
        return [artifact for artifact in self.artifacts if artifact.identifier == identifier][0]
