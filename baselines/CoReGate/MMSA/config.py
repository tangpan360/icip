import json
import os
import random
from pathlib import Path

from easydict import EasyDict as edict


def get_config_regression(model_name: str, dataset_name: str, config_file: str = "") -> dict:
    if config_file == "":
        config_file = Path(__file__).parent / "config" / "config_regression.json"
    with open(config_file, "r") as f:
        config_all = json.load(f)

    model_common_args = config_all[model_name]["commonParams"]
    model_dataset_args = config_all[model_name]["datasetParams"][dataset_name]
    dataset_args = config_all["datasetCommonParams"][dataset_name]

    # use aligned feature if the model requires it, otherwise use unaligned feature
    if model_common_args["need_data_aligned"] and "aligned" in dataset_args:
        dataset_args = dataset_args["aligned"]
    else:
        dataset_args = dataset_args["unaligned"]

    config = {}
    config["model_name"] = model_name
    config["dataset_name"] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    config["featurePath"] = os.path.join(config_all["datasetCommonParams"]["dataset_root_dir"], config["featurePath"])
    return edict(config)


def get_config_tune(
    model_name: str,
    dataset_name: str,
    config_file: str = "",
    random_choice: bool = True,
) -> dict:
    if config_file == "":
        config_file = Path(__file__).parent / "config" / "config_tune.json"
    with open(config_file, "r") as f:
        config_all = json.load(f)

    model_common_args = config_all[model_name]["commonParams"]
    model_dataset_args = (
        config_all[model_name]["datasetParams"][dataset_name] if "datasetParams" in config_all[model_name] else {}
    )
    model_debug_args = config_all[model_name]["debugParams"]
    dataset_args = config_all["datasetCommonParams"][dataset_name]
    dataset_args = (
        dataset_args["aligned"]
        if (model_common_args["need_data_aligned"] and "aligned" in dataset_args)
        else dataset_args["unaligned"]
    )

    if random_choice:
        for item in model_debug_args["d_paras"]:
            if isinstance(model_debug_args[item], list):
                model_debug_args[item] = random.choice(model_debug_args[item])
            elif isinstance(model_debug_args[item], dict):
                for k, v in model_debug_args[item].items():
                    model_debug_args[item][k] = random.choice(v)

    config = {}
    config["model_name"] = model_name
    config["dataset_name"] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    config.update(model_debug_args)
    config["featurePath"] = os.path.join(config_all["datasetCommonParams"]["dataset_root_dir"], config["featurePath"])
    return edict(config)


def get_config_all(config_file: str) -> dict:
    if config_file == "regression":
        config_file = Path(__file__).parent / "config" / "config_regression.json"
    elif config_file == "tune":
        config_file = Path(__file__).parent / "config" / "config_tune.json"
    else:
        raise ValueError("config_file should be 'regression' or 'tune'")
    with open(config_file, "r") as f:
        config_all = json.load(f)
    return edict(config_all)

