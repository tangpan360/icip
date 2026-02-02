import logging

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from ...utils import MetricsTop, dict_to_str

logger = logging.getLogger("MMSA")


class CENET_TMCC:
    """
    Trainer for CENET_TMCC.

    Loss:
      L = L_task(M, y) + aux_fusion_weight * L_task(y_f, y)

    Notes:
    - aux_fusion_weight defaults to 0.0 (no extra loss), so you can test quickly.
    """

    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == "regression" else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        # mild regularization: auxiliary fusion head (can be overridden in config)
        self.aux_fusion_weight = float(getattr(args, "aux_fusion_weight", 0.1))
        self.max_grad_norm = float(getattr(args, "max_grad_norm", 2.0))
        self.adam_epsilon = float(getattr(args, "adam_epsilon", 1e-8))

    def _prep_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if self.args.train_mode == "classification":
            return labels.view(-1).long()
        return labels.view(-1, 1)

    def do_train(self, model, dataloader, return_epoch_results=False):
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            eps=self.adam_epsilon,
        )

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {"train": [], "valid": [], "test": []}

        min_or_max = "min" if self.args.KeyEval in ["Loss"] else "max"
        best_valid = 1e8 if min_or_max == "min" else 0

        while True:
            epochs += 1
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0

            with tqdm(dataloader["train"]) as td:
                for batch_data in td:
                    vision = batch_data["vision"].to(self.args.device)
                    audio = batch_data["audio"].to(self.args.device)
                    text = batch_data["text"].to(self.args.device)
                    labels = self._prep_labels(batch_data["labels"]["M"].to(self.args.device))

                    # Use true lengths for unaligned A/V if available; improves masking/attention pooling.
                    if not getattr(self.args, "need_data_aligned", False):
                        audio_l = batch_data.get("audio_lengths", None)
                        vision_l = batch_data.get("vision_lengths", None)
                        if audio_l is not None and vision_l is not None:
                            audio_in = (audio, audio_l)
                            vision_in = (vision, vision_l)
                        else:
                            audio_in = audio
                            vision_in = vision
                    else:
                        audio_in = audio
                        vision_in = vision

                    optimizer.zero_grad()
                    outs = model(text, audio_in, vision_in)
                    pred = outs["M"]
                    loss = self.criterion(pred, labels)

                    if self.aux_fusion_weight > 0 and "y_f" in outs:
                        loss = loss + self.aux_fusion_weight * self.criterion(outs["y_f"], labels)

                    loss.backward()
                    if self.max_grad_norm and self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    optimizer.step()

                    train_loss += loss.item()
                    y_pred.append(pred.detach().cpu())
                    y_true.append(labels.detach().cpu())

            train_loss = train_loss / len(dataloader["train"])
            pred_all, true_all = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred_all, true_all)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >> loss: {round(train_loss, 4)} {dict_to_str(train_results)}"
            )

            val_results = self.do_test(model, dataloader["valid"], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == "min" else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results["train"].append(train_results)
                epoch_results["valid"].append(val_results)
                test_results = self.do_test(model, dataloader["test"], mode="TEST")
                epoch_results["test"].append(test_results)

            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data["vision"].to(self.args.device)
                    audio = batch_data["audio"].to(self.args.device)
                    text = batch_data["text"].to(self.args.device)
                    labels = self._prep_labels(batch_data["labels"]["M"].to(self.args.device))

                    if not getattr(self.args, "need_data_aligned", False):
                        audio_l = batch_data.get("audio_lengths", None)
                        vision_l = batch_data.get("vision_lengths", None)
                        if audio_l is not None and vision_l is not None:
                            audio_in = (audio, audio_l)
                            vision_in = (vision, vision_l)
                        else:
                            audio_in = audio
                            vision_in = vision
                    else:
                        audio_in = audio
                        vision_in = vision

                    outs = model(text, audio_in, vision_in)
                    pred = outs["M"]
                    loss = self.criterion(pred, labels)
                    if self.aux_fusion_weight > 0 and "y_f" in outs:
                        loss = loss + self.aux_fusion_weight * self.criterion(outs["y_f"], labels)

                    eval_loss += loss.item()
                    y_pred.append(pred.detach().cpu())
                    y_true.append(labels.detach().cpu())

        eval_loss = eval_loss / len(dataloader)
        pred_all, true_all = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred_all, true_all)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")
        return eval_results

