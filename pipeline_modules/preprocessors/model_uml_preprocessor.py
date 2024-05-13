import xml.etree.ElementTree as ET

from cache.cache_manager import CacheManager
from .preprocessor import Preprocessor
from ..knowledge import Element, Artifact
from ..module import ModuleConfiguration


class ModelUMLPreprocessor(Preprocessor):
    __configuration: ModuleConfiguration

    def __init__(self, configuration: ModuleConfiguration):
        self.use_prefix = configuration.args.setdefault('use_prefix', True)
        self.include_usages = configuration.args.setdefault('include_usages', True)
        self.include_operations = configuration.args.setdefault('include_operations', True)
        self.include_interface_realizations = configuration.args.setdefault('include_interface_realizations', True)
        self.__configuration = configuration

    def __get_cached(self, artifact: Element) -> list[Element]:
        # TODO: refactor out caching to another class (same as simple_text_preprocessor)

        data = CacheManager.get_cache().get(configuration=self.__configuration, input_key=artifact.to_json())
        elements: list[Element] = list()
        parent_mapping: dict[str, str] = {}

        for element_dict in data:
            element = Element.element_from_dict(element_dict)
            elements.append(element)
            parent_mapping[element.identifier] = element_dict["parent"]

        for element in elements:
            if parent_mapping[element.identifier] is not None:
                element.parent = [e for e in elements if e.identifier == parent_mapping[element.identifier]][0]
            else:
                element.parent = None
        return elements

    def preprocess(self, artifact: Artifact) -> list[Element]:
        elements = self.__get_cached(artifact)
        if elements:
            return elements
        elements.append(artifact)

        # TODO: parse namespace instead of hardcoding
        ns = {
            "xmi": "http://www.omg.org/spec/XMI/20131001",
            "uml": "http://www.eclipse.org/uml2/5.0.0/UML"
        }

        root = ET.fromstring(artifact.content)
        i = 0
        for packagedElement in root.findall('packagedElement'):
            element_type = packagedElement.get(f'{{{ns["xmi"]}}}type')

            if not self.use_prefix:
                # remove the prefix, such as "uml:" from "uml:Component"
                substrings = element_type.split(':', 1)
                element_type = substrings[1] if len(substrings) > 1 else element_type

            element_id = packagedElement.get(f'{{{ns["xmi"]}}}id')
            element_name = packagedElement.get('name')
            content = f'Type: {element_type}, Name: {element_name}'
            if self.include_interface_realizations:
                for interface_realization in packagedElement.findall('interfaceRealization'):
                    supplier_id = interface_realization.get('supplier')
                    supplier = root.find('.//*[@xmi:id="%s"]' % supplier_id, ns)
                    supplier_name = supplier.get('name')
                    content = content + f'\n Interface Realization: {supplier_name}'
            if self.include_operations:
                for operation in packagedElement.findall('ownedOperation'):
                    operation_name = operation.get('name')
                    content = content + f"\n Operation: {operation_name}"
            if self.include_usages:
                for usage in packagedElement.findall('packagedElement'):
                    supplier_id = usage.get('supplier')
                    supplier = root.find('.//*[@xmi:id="%s"]' % supplier_id, ns)
                    supplier_name = supplier.get('name')
                    content = content + f"\n Uses: {supplier_name}"

            element = Element(identifier=artifact.identifier + "$" + str(i) + "$" + element_id,
                              type=artifact.type,
                              content=content,
                              parent=artifact,
                              granularity=1,
                              compare=element_type == 'uml:Component' or element_type == 'Component')
            elements.append(element)
            i = i+1
            print(element.identifier)

        for element in elements:
            CacheManager.get_cache().put(configuration=self.__configuration, input=artifact.to_json(),
                                         data=element.to_dict())
        return elements
