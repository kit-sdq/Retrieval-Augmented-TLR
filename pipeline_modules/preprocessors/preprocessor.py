from typing import Protocol
from ..module import ModuleConfiguration
from ..knowledge import Artifact, Element


class Preprocessor(Protocol):
    def preprocess(self, artifact: Artifact) -> list[Element]:
        ...


class PreprocessorBuilder:
    from .simple_text_preprocessor import SimpleTextPreprocessor
    from .sentence_splitter_preprocessor import SentenceSplitterPreprocessor
    from .code_chunking_preprocessor import CodeChunkingPreprocessor
    from .code_method_preprocessor import CodeMethodPreprocessor
    from .model_uml_preprocessor import ModelUMLPreprocessor
    from .line_splitter_preprocessor import LineSplitterPreprocessor

    PREPROCESSORS = {
        'simple': SimpleTextPreprocessor,
        'sentence': SentenceSplitterPreprocessor,
        'code_chunking': CodeChunkingPreprocessor,
        'code_method': CodeMethodPreprocessor,
        'model_uml': ModelUMLPreprocessor,
        'line': LineSplitterPreprocessor
    }

    def build_preprocessor(self, configuration: ModuleConfiguration) -> Preprocessor:
        return self.PREPROCESSORS[configuration.name](configuration)
