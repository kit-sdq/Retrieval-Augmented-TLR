import re

from project.pipeline_modules.element_store.element_store import ElementStore
from project.pipeline_modules.knowledge import Element


class ContextProvider:
    source_store: ElementStore
    target_store: ElementStore

    def __init__(self, source_store: ElementStore, target_store: ElementStore):
        self.source_store = source_store
        self.target_store = target_store

    def neighbouring_sibling_context(self, is_source: bool, element: Element, pre: int, post: int) -> (str, str):
        """Provides the content of siblings of the given element.
        The contents are concatenated into a "before" and "after" str."""
        if pre == 0 and post == 0:
            return "", ""

        store = self.source_store if is_source else self.target_store

        siblings = [x.element for x in store.get_by_parent_id(element.parent.identifier)]
        # Simple natural sort, might break for edge cases
        siblings = sorted(siblings, key=lambda x: [int(y) if y.isdigit() else y.lower() for y in re.split("(\\d+)", x.identifier)])
        index = [index for index, value in enumerate(siblings) if value.identifier == element.identifier][0]

        pre_start = max(index-pre, 0)
        pre_context = siblings[pre_start:index]
        post_end = min(index+post, len(siblings)-1)
        post_context = siblings[index+1:post_end+1]

        pre_text = "\n".join(element.content for element in pre_context)
        post_text = "\n".join(element.content for element in post_context)

        return pre_text, post_text

