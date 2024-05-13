from pathlib import Path

from .artifact_provider import ArtifactProvider
from ..knowledge import Artifact
from ..module import ModuleConfiguration


class TextArtifactProvider(ArtifactProvider):
    """Provides artifacts with their content being the texts from utf-8 encoded text files located in the same folder."""
    __configuration: ModuleConfiguration
    __artifacts: list[Artifact]
    __path: str

    def __init__(self, configuration: ModuleConfiguration):
        self.__configuration = configuration
        self.__path = self.__configuration.args["path"]

        self.__artifacts = list()
        for file_path in Path(self.__path).iterdir():
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
