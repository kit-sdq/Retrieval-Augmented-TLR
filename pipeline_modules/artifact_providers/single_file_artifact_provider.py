from pathlib import Path

from .artifact_provider import ArtifactProvider
from ..knowledge import Artifact
from ..module import ModuleConfiguration


class SingleFileArtifactProvider(ArtifactProvider):
    """Provides a single artifact with its content being the complete text from an utf-8 encoded text file."""
    __configuration: ModuleConfiguration
    __artifacts: list[Artifact]
    __path: str

    def __init__(self, configuration: ModuleConfiguration):
        self.__configuration = configuration
        self.__path = self.__configuration.args["path"]

        self.__artifacts = list()
        file_path = Path(self.__path)
        print(file_path)
        file = open(file_path, 'r', encoding="utf-8")
        content = file.read()
        identifier = file_path.stem
        artifact = Artifact(identifier=identifier, type=self.__configuration.args["artifact_type"], content=content)
        self.__artifacts.append(artifact)

    def get_all_artifacts(self) -> list[Artifact]:
        return self.__artifacts

    def get_artifact(self, identifier: str) -> Artifact:
        return [artifact for artifact in self.__artifacts if artifact.identifier == identifier][0]
