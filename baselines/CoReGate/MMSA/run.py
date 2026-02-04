import errno
import gc
import json
import logging
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict

from .config import get_config_regression, get_config_tune
from .data_loader import MMDataLoader
from .models import AMIO
from .trains import ATIO
from .utils import assign_gpu, count_parameters, setup_seed

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

SUPPORTED_MODELS = ["CoReGate"]
SUPPORTED_DATASETS = ["MOSI", "MOSEI"]

logger = logging.getLogger("MMSA")


def _set_logger(log_dir, model_name, dataset_name, verbose_level):
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger("MMSA")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter("%(asctime)s - %(name)s [%(levelname)s] - %(message)s")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter("%(name)s - %(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    return logger


def MMSA_run(
    model_name: str,
    dataset_name: str,
    config_file: str = None,
    config: dict = None,
    seeds: list = [],
    is_tune: bool = False,
    tune_times: int = 50,
    custom_feature: str = None,
    feature_T: str = None,
    feature_A: str = None,
    feature_V: str = None,
    gpu_ids: list = [0],
    num_workers: int = 4,
    verbose_level: int = 1,
    model_save_dir: str = Path().home() / "MMSA" / "saved_models",
    res_save_dir: str = Path().home() / "MMSA" / "results",
    log_dir: str = Path().home() / "MMSA" / "logs",
):
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    if config_file is not None and str(config_file).strip() == "":
        config_file = None

    if config_file is not None:
        config_file = Path(config_file)
    else:
        config_file = Path(__file__).parent / "config" / ("config_tune.json" if is_tune else "config_regression.json")
    if not config_file.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)

    if model_save_dir is None:
        model_save_dir = Path.home() / "MMSA" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir is None:
        res_save_dir = Path.home() / "MMSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir is None:
        log_dir = Path.home() / "MMSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    _set_logger(log_dir, model_name, dataset_name, verbose_level)
    logger.info("======================================== Program Start ========================================")

    if is_tune:
        logger.info(f"Tuning with seed {seeds[0]}")
        initial_args = get_config_tune(model_name, dataset_name, config_file)
        initial_args["model_save_path"] = Path(model_save_dir) / f"{initial_args['model_name']}-{initial_args['dataset_name']}.pth"
        initial_args["device"] = assign_gpu(gpu_ids)
        initial_args["train_mode"] = "regression"
        initial_args["custom_feature"] = custom_feature
        initial_args["feature_T"] = feature_T
        initial_args["feature_A"] = feature_A
        initial_args["feature_V"] = feature_V
        torch.cuda.set_device(initial_args["device"])

        res_save_dir = Path(res_save_dir) / "tune"
        res_save_dir.mkdir(parents=True, exist_ok=True)
        has_debuged = []
        csv_file = res_save_dir / f"{dataset_name}-{model_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
            for i in range(len(df)):
                has_debuged.append([df.loc[i, k] for k in initial_args["d_paras"]])

        for i in range(tune_times):
            args = edict(**initial_args)
            random.seed(time.time())
            new_args = get_config_tune(model_name, dataset_name, config_file)
            args.update(new_args)
            if config:
                if config.get("model_name"):
                    assert config["model_name"] == args["model_name"]
                args.update(config)
            args["cur_seed"] = i + 1
            logger.info(f"{'-' * 30} Tuning [{i + 1}/{tune_times}] {'-' * 30}")
            logger.info(f"Args: {args}")
            cur_param = [args[k] for k in args["d_paras"]]
            if cur_param in has_debuged:
                logger.info("This set of parameters has been run. Skip.")
                time.sleep(1)
                continue
            setup_seed(seeds[0])
            result = _run(args, num_workers, is_tune)
            has_debuged.append(cur_param)
            if Path(csv_file).is_file():
                df2 = pd.read_csv(csv_file)
            else:
                df2 = pd.DataFrame(columns=[k for k in args.d_paras] + [k for k in result.keys()])
            res = [args[c] for c in args.d_paras]
            for col in result.keys():
                res.append(result[col])
            df2.loc[len(df2)] = res
            df2.to_csv(csv_file, index=None)
            logger.info(f"Results saved to {csv_file}.")
        return

    args = get_config_regression(model_name, dataset_name, config_file)
    args["model_save_path"] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
    args["device"] = assign_gpu(gpu_ids)
    args["train_mode"] = "regression"
    args["custom_feature"] = custom_feature
    args["feature_T"] = feature_T
    args["feature_A"] = feature_A
    args["feature_V"] = feature_V
    if config:
        if config.get("model_name"):
            assert config["model_name"] == args["model_name"]
        args.update(config)

    torch.cuda.set_device(args["device"])
    logger.info("Running with args:")
    logger.info(args)
    logger.info(f"Seeds: {seeds}")

    res_save_dir = Path(res_save_dir) / "normal"
    res_save_dir.mkdir(parents=True, exist_ok=True)
    model_results = []
    for i, seed in enumerate(seeds):
        setup_seed(seed)
        args["cur_seed"] = i + 1
        logger.info(f"{'-' * 30} Running with seed {seed} [{i + 1}/{len(seeds)}] {'-' * 30}")
        result = _run(args, num_workers, is_tune)
        logger.info(f"Result for seed {seed}: {result}")
        model_results.append(result)

    criterions = list(model_results[0].keys())
    csv_file = res_save_dir / f"{dataset_name}.csv"
    if csv_file.is_file():
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    res = [model_name]
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values) * 100, 2)
        std = round(np.std(values) * 100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    df.to_csv(csv_file, index=None)
    logger.info(f"Results saved to {csv_file}.")


def _run(args, num_workers=4, is_tune=False):
    dataloader = MMDataLoader(args, num_workers)
    model = AMIO(args).to(args["device"])
    logger.info(f"The model has {count_parameters(model)} trainable parameters")
    trainer = ATIO().getTrain(args)
    trainer.do_train(model, dataloader, return_epoch_results=False)
    assert Path(args["model_save_path"]).exists()
    model.load_state_dict(torch.load(args["model_save_path"]))
    model.to(args["device"])
    results = trainer.do_test(model, dataloader["test"], mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)
    return results


def MMSA_test(config: dict | str, weights_path: str, feature_path: str, gpu_id: int = 0):
    if type(config) == str or type(config) == Path:
        config = Path(config)
        with open(config, "r") as f:
            args = json.load(f)
    elif type(config) == dict or type(config) == edict:
        args = config
    else:
        raise ValueError(f"'config' should be string or dict, not {type(config)}")
    args["train_mode"] = "regression"
    device = torch.device("cpu") if gpu_id < 0 else torch.device(f"cuda:{gpu_id}")
    args["device"] = device
    with open(feature_path, "rb") as f:
        feature = pickle.load(f)
    args["feature_dims"] = [feature["text"].shape[1], feature["audio"].shape[1], feature["vision"].shape[1]]
    args["seq_lens"] = [feature["text"].shape[0], feature["audio"].shape[0], feature["vision"].shape[0]]
    model = AMIO(args)
    model.load_state_dict(torch.load(weights_path), strict=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        if args.get("use_bert", None):
            if type(text := feature["text_bert"]) == np.ndarray:
                text = torch.from_numpy(text).float()
        else:
            if type(text := feature["text"]) == np.ndarray:
                text = torch.from_numpy(text).float()
        if type(audio := feature["audio"]) == np.ndarray:
            audio = torch.from_numpy(audio).float()
        if type(vision := feature["vision"]) == np.ndarray:
            vision = torch.from_numpy(vision).float()
        text = text.unsqueeze(0).to(device)
        audio = audio.unsqueeze(0).to(device)
        vision = vision.unsqueeze(0).to(device)
        if args.get("need_normalized", None):
            audio = torch.mean(audio, dim=1, keepdims=True)
            vision = torch.mean(vision, dim=1, keepdims=True)
        output = model(text, audio, vision)
        if type(output) == dict:
            output = output["M"]
    return output.cpu().detach().numpy()[0][0]

