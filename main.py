import json
from typing import Any

from cache.cache_manager import CacheManager
from controller import Controller
from pipeline_modules.module import ModuleConfiguration, PipelineConfiguration

MODULE_TYPES = {
    "classifier": "classifier",
    "embedding_creator": "embedding_creator",
    "result_aggregator": "result_aggregator",
    "target_store": "element_store",
    "source_store": "element_store",
    "target_artifact_provider": "artifact_provider",
    "source_artifact_provider": "artifact_provider",
    "target_preprocessor": "preprocessor",
    "source_preprocessor": "preprocessor"
}


def __load_module_config(name: str, arguments: Any) -> ModuleConfiguration:
    config = ModuleConfiguration()
    config.type = MODULE_TYPES[name]
    config.name = arguments[name]["name"]
    config.args = arguments[name]["args"]
    return config


def load_config(path: str):
    # TODO: implement in separate file/class
    file = open(path)
    arguments = json.load(file)
    config = PipelineConfiguration()
    config.classifier = __load_module_config("classifier", arguments)
    config.embedding_creator = __load_module_config("embedding_creator", arguments)
    config.result_aggregator = __load_module_config("result_aggregator", arguments)
    config.target_store = __load_module_config("target_store", arguments)
    config.source_store = __load_module_config("source_store", arguments)
    config.target_artifact_provider = __load_module_config("target_artifact_provider", arguments)
    config.source_artifact_provider = __load_module_config("source_artifact_provider", arguments)
    config.target_preprocessor = __load_module_config("target_preprocessor", arguments)
    config.source_preprocessor = __load_module_config("source_preprocessor", arguments)

    return config


if __name__ == '__main__':
    from evaluation import calculate_f1

    do_reverse = False

    results = list()
    reversed_results = list()

    req_code_paths = [
        # (source_provider, target_provider, embedding, source_store, target_store, ground_truth, name)
        ("./datasets/iTrust/UC/", "./datasets/iTrust/CC/", "./storage/iTrust/embeddings/", "./storage/iTrust/",
         "./storage/iTrust/", "./datasets/iTrust/UC2JAVA.csv", "iTrust")
    ]

    path_tuples = req_code_paths  # sad_sam_paths #req_code_paths
    for tuple in path_tuples:
        # pipeline_config = load_config("configuration_sad_sam.json")
        # pipeline_config = load_config("configuration_sad_code.json")
        pipeline_config = load_config("configuration_req_code.json")
        # pipeline_config = load_config("configuration_test.json")

        # Override path to datasets
        pipeline_config.source_artifact_provider.args["path"] = tuple[0]
        pipeline_config.target_artifact_provider.args["path"] = tuple[1]
        pipeline_config.embedding_creator.args["path"] = tuple[2]
        pipeline_config.source_store.args["path"] = tuple[3]
        pipeline_config.target_store.args["path"] = tuple[4]

        CacheManager(database_name="cache", folder_path=pipeline_config.target_store.args["path"])
        controller = Controller(pipeline_configuration=pipeline_config)
        links = controller.run()

        print(f"RESULTS: {tuple[6]}")
        results.append(calculate_f1([(link.source, link.target) for link in links], tuple[5], False, False))

        # REVERSE
        if do_reverse:
            new_source_artifact_provider = pipeline_config.target_artifact_provider
            pipeline_config.target_artifact_provider = pipeline_config.source_artifact_provider
            pipeline_config.source_artifact_provider = new_source_artifact_provider

            new_source_preprocessor = pipeline_config.target_preprocessor
            pipeline_config.target_preprocessor = pipeline_config.source_preprocessor
            pipeline_config.source_preprocessor = new_source_preprocessor

            # IMPORTANT: source and target store should have the same general configuration
            new_source_store = pipeline_config.target_store
            pipeline_config.target_store = pipeline_config.source_store
            pipeline_config.source_store = new_source_store

            CacheManager(database_name="cache", folder_path=pipeline_config.target_store.args["path"])
            controller = Controller(pipeline_configuration=pipeline_config)
            links = controller.run()

            print(f"RESULTS: {tuple[6]}-reversed")
            reversed_results.append(calculate_f1([(link.source, link.target) for link in links], tuple[5], False, True))

    csv_string = ""
    for result in results:
        csv_string += f"{round(result[0], 3)},{round(result[1], 3)},{round(result[2], 3)},"
    print("CSV:")
    print(csv_string)

    if do_reverse:
        csv_string_reversed = ""
        for result in reversed_results:
            csv_string_reversed += f"{round(result[0], 3)},{round(result[1], 3)},{round(result[2], 3)},"
        print("CSV_reversed:")
        print(csv_string_reversed)
