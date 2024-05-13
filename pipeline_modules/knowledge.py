import typing
import json


class Knowledge:
    identifier: str
    type: str
    content: object

    def __init__(self, identifier: str, type: str, content: object):
        self.identifier = identifier
        self.type = type
        self.content = content


class Element(Knowledge):
    parent: Knowledge | None
    granularity: int  # 0 represents the most coarse granularity.
    content: str
    compare: bool

    def __init__(self, identifier: str, type: str, content: str, granularity: int, parent: Knowledge | None, compare: bool = True):
        super().__init__(identifier=identifier, type=type, content=content)
        self.parent = parent
        self.granularity = granularity
        self.compare = compare

    def original_artifact(self) -> "Element":
        element = self
        while element.granularity > 0:
            element = element.parent
        return element

    def to_dict(self) -> dict:
        """Transforms the Element into a json compatible dictionary representation.
        Parents are referred to by their identifier"""
        dictionary = dict()
        dictionary["identifier"] = self.identifier
        dictionary["type"] = self.type
        dictionary["content"] = self.content
        dictionary["granularity"] = self.granularity
        dictionary["compare"] = self.compare

        if self.granularity != 0:
            dictionary["parent"] = self.parent.identifier
        else:
            dictionary["parent"] = None

        return dictionary

    def to_json(self) -> str:
        """Transforms the Element into a json representation. Parents are referred to by their identifier"""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def element_from_dict(cls, element: dict) -> 'Element':
        """Creates Elements from a dictionary. parent is set to None"""
        return Element(identifier=element["identifier"],
                       type=element["type"],
                       content=element["content"],
                       granularity=element["granularity"],
                       parent=None,
                       compare=element["compare"])


class Artifact(Element):
    def __init__(self, identifier: str, type: str, content: object):
        super().__init__(identifier=identifier, type=type, content=content, granularity=0, parent=None, compare=False)
