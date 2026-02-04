import logging
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ["MMDataLoader"]

logger = logging.getLogger("MMSA")


class MMDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.mode = mode
        self.args = args
        DATASET_MAP = {
            "mosi": self.__init_mosi,
            "mosei": self.__init_mosei,
            "sims": self.__init_sims,
            "simsv2": self.__init_simsv2,
        }
        DATASET_MAP[args["dataset_name"]]()

    def __init_mosi(self):
        if self.args["custom_feature"]:
            with open(self.args["custom_feature"], "rb") as f:
                data = pickle.load(f)
        else:
            with open(self.args["featurePath"], "rb") as f:
                data = pickle.load(f)

        if self.args.get("use_bert", None):
            self.text = data[self.mode]["text_bert"].astype(np.float32)
            self.args["feature_dims"][0] = 768
        else:
            self.text = data[self.mode]["text"].astype(np.float32)
            self.args["feature_dims"][0] = self.text.shape[2]
        self.audio = data[self.mode]["audio"].astype(np.float32)
        self.args["feature_dims"][1] = self.audio.shape[2]
        self.vision = data[self.mode]["vision"].astype(np.float32)
        self.args["feature_dims"][2] = self.vision.shape[2]
        self.raw_text = data[self.mode]["raw_text"]
        self.ids = data[self.mode]["id"]

        # Override with custom modality features
        if self.args["feature_T"]:
            with open(self.args["feature_T"], "rb") as f:
                data_T = pickle.load(f)
            if self.args.get("use_bert", None):
                self.text = data_T[self.mode]["text_bert"].astype(np.float32)
                self.args["feature_dims"][0] = 768
            else:
                self.text = data_T[self.mode]["text"].astype(np.float32)
                self.args["feature_dims"][0] = self.text.shape[2]
        if self.args["feature_A"]:
            with open(self.args["feature_A"], "rb") as f:
                data_A = pickle.load(f)
            self.audio = data_A[self.mode]["audio"].astype(np.float32)
            self.args["feature_dims"][1] = self.audio.shape[2]
        if self.args["feature_V"]:
            with open(self.args["feature_V"], "rb") as f:
                data_V = pickle.load(f)
            self.vision = data_V[self.mode]["vision"].astype(np.float32)
            self.args["feature_dims"][2] = self.vision.shape[2]

        self.labels = {"M": np.array(data[self.mode]["regression_labels"]).astype(np.float32)}
        if "sims" in self.args["dataset_name"]:
            for m in "TAV":
                self.labels[m] = data[self.mode]["regression" + "_labels_" + m].astype(np.float32)

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args["need_data_aligned"]:
            if self.args["feature_A"]:
                self.audio_lengths = list(data_A[self.mode]["audio_lengths"])
            else:
                self.audio_lengths = data[self.mode]["audio_lengths"]
            if self.args["feature_V"]:
                self.vision_lengths = list(data_V[self.mode]["vision_lengths"])
            else:
                self.vision_lengths = data[self.mode]["vision_lengths"]
        self.audio[self.audio == -np.inf] = 0

        if self.args.get("data_missing"):
            self.text_m, self.text_length, self.text_mask, self.text_missing_mask = self.generate_m(
                self.text[:, 0, :],
                self.text[:, 1, :],
                None,
                self.args.missing_rate[0],
                self.args.missing_seed[0],
                mode="text",
            )
            Input_ids_m = np.expand_dims(self.text_m, 1)
            Input_mask = np.expand_dims(self.text_mask, 1)
            Segment_ids = np.expand_dims(self.text[:, 2, :], 1)
            self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1)

            if self.args["need_data_aligned"]:
                self.audio_lengths = np.sum(self.text[:, 1, :], axis=1, dtype=np.int32)
                self.vision_lengths = np.sum(self.text[:, 1, :], axis=1, dtype=np.int32)

            self.audio_m, self.audio_length, self.audio_mask, self.audio_missing_mask = self.generate_m(
                self.audio,
                None,
                self.audio_lengths,
                self.args.missing_rate[1],
                self.args.missing_seed[1],
                mode="audio",
            )
            self.vision_m, self.vision_length, self.vision_mask, self.vision_missing_mask = self.generate_m(
                self.vision,
                None,
                self.vision_lengths,
                self.args.missing_rate[2],
                self.args.missing_seed[2],
                mode="vision",
            )

        if self.args.get("need_normalized"):
            self.__normalize()

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __init_simsv2(self):
        return self.__init_mosi()

    def generate_m(self, modality, input_mask, input_len, missing_rate, missing_seed, mode="text"):
        if mode == "text":
            input_len = np.argmin(input_mask, axis=1)
        elif mode == "audio" or mode == "vision":
            input_mask = np.array(
                [np.array([1] * length + [0] * (modality.shape[1] - length)) for length in input_len]
            )
        np.random.seed(missing_seed)
        missing_mask = (np.random.uniform(size=input_mask.shape) > missing_rate) * input_mask
        assert missing_mask.shape == input_mask.shape

        if mode == "text":
            for i, instance in enumerate(missing_mask):
                instance[0] = instance[input_len[i] - 1] = 1
            modality_m = missing_mask * modality + (100 * np.ones_like(modality)) * (input_mask - missing_mask)
        elif mode == "audio" or mode == "vision":
            modality_m = missing_mask.reshape(modality.shape[0], modality.shape[1], 1) * modality

        return modality_m, input_len, input_mask, missing_mask

    def __normalize(self):
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels["M"])

    def get_seq_len(self):
        if "use_bert" in self.args and self.args["use_bert"]:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def __getitem__(self, index):
        sample = {
            "raw_text": self.raw_text[index],
            "text": torch.Tensor(self.text[index]),
            "audio": torch.Tensor(self.audio[index]),
            "vision": torch.Tensor(self.vision[index]),
            "index": index,
            "id": self.ids[index],
            "labels": {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
        }
        if not self.args["need_data_aligned"]:
            sample["audio_lengths"] = self.audio_lengths[index]
            sample["vision_lengths"] = self.vision_lengths[index]
        if self.args.get("data_missing"):
            sample["text_m"] = torch.Tensor(self.text_m[index])
            sample["text_missing_mask"] = torch.Tensor(self.text_missing_mask[index])
            sample["audio_m"] = torch.Tensor(self.audio_m[index])
            sample["audio_lengths"] = self.audio_lengths[index]
            sample["audio_mask"] = self.audio_mask[index]
            sample["audio_missing_mask"] = torch.Tensor(self.audio_missing_mask[index])
            sample["vision_m"] = torch.Tensor(self.vision_m[index])
            sample["vision_lengths"] = self.vision_lengths[index]
            sample["vision_mask"] = self.vision_mask[index]
            sample["vision_missing_mask"] = torch.Tensor(self.vision_missing_mask[index])
        return sample


def MMDataLoader(args, num_workers):
    datasets = {
        "train": MMDataset(args, mode="train"),
        "valid": MMDataset(args, mode="valid"),
        "test": MMDataset(args, mode="test"),
    }
    if "seq_lens" in args:
        args["seq_lens"] = datasets["train"].get_seq_len()
    return {
        ds: DataLoader(datasets[ds], batch_size=args["batch_size"], num_workers=num_workers, shuffle=True)
        for ds in datasets.keys()
    }

