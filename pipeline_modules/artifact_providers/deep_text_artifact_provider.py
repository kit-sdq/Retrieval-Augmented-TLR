from pathlib import Path

from .artifact_provider import ArtifactProvider
from ..knowledge import Artifact
from ..module import ModuleConfiguration


class DeepTextArtifactProvider(ArtifactProvider):
    """Provides artifacts with their content being the texts from utf-8 encoded text files.
    Searches subdirectories for files.
    Needs absolute file path to correctly assign identifiers."""
    __configuration: ModuleConfiguration
    __artifacts: list[Artifact]
    __path: str
    __extensions: list[str]
    __type: str

    def __init__(self, configuration: ModuleConfiguration):
        self.__configuration = configuration
        self.__path = self.__configuration.args["path"]
        self.__extensions = self.__configuration.args["extensions"]
        self.__type = self.__configuration.args["artifact_type"]

        self.__artifacts = list()
        for extension in self.__extensions:
            for file_path in Path(self.__path).rglob("*" + extension):
                print(file_path)
                file = open(file_path, 'r', encoding="utf-8")
                content = file.read()
                identifier = file_path.as_posix().removeprefix(self.__path).removeprefix("/")
                artifact = Artifact(identifier=identifier, type=self.__type, content=content)
                self.__artifacts.append(artifact)

    def get_all_artifacts(self) -> list[Artifact]:
        return self.__artifacts

    def get_artifact(self, identifier: str) -> Artifact:
        return [artifact for artifact in self.__artifacts if artifact.identifier == identifier][0]
