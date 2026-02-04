import logging

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from ...utils import MetricsTop, dict_to_str

logger = logging.getLogger("MMSA")


class CoReGate:
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() if args.train_mode == "regression" else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.branch_loss_weight = float(getattr(args, "branch_loss_weight", 0.1))
        self.max_grad_norm = float(getattr(args, "max_grad_norm", 2.0))
        self.adam_epsilon = float(getattr(args, "adam_epsilon", 1e-8))

    def _prep_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if self.args.train_mode == "classification":
            return labels.view(-1).long()
        return labels.view(-1, 1)

    def _pack_av_inputs(self, batch_data):
        vision = batch_data["vision"].to(self.args.device)
        audio = batch_data["audio"].to(self.args.device)
        text = batch_data["text"].to(self.args.device)
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
        return text, audio_in, vision_in

    @staticmethod
    def _zero_like_av(x):
        if isinstance(x, (tuple, list)) and len(x) >= 1:
            t = x[0]
            rest = x[1:]
            return (torch.zeros_like(t), *rest)
        return torch.zeros_like(x)

    def do_train(self, model, dataloader, return_epoch_results=False):
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            eps=self.adam_epsilon,
        )
        epochs, best_epoch = 0, 0
        min_or_max = "min" if self.args.KeyEval in ["Loss"] else "max"
        best_valid = 1e8 if min_or_max == "min" else 0

        while True:
            epochs += 1
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0

            with tqdm(dataloader["train"]) as td:
                for batch_data in td:
                    labels = self._prep_labels(batch_data["labels"]["M"].to(self.args.device))
                    text, audio_in, vision_in = self._pack_av_inputs(batch_data)

                    optimizer.zero_grad()
                    outs = model(text, audio_in, vision_in)
                    pred = outs["M"]
                    loss = self.criterion(pred, labels)

                    if self.branch_loss_weight > 0:
                        for k in ("y_t", "y_ta", "y_tv"):
                            if k in outs:
                                loss = loss + self.branch_loss_weight * self.criterion(outs[k], labels)

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

            if epochs - best_epoch >= self.args.early_stop:
                return None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        eval_missing = str(getattr(self.args, "eval_missing", "none")).strip().lower()
        eval_disable_gating = bool(getattr(self.args, "eval_disable_gating", False))
        if mode != "TEST":
            eval_missing = "none"
            eval_disable_gating = False
        if eval_missing not in ("none", "a", "v", "av", "audio", "vision"):
            raise ValueError(f"Unsupported eval_missing: {eval_missing}")

        inner_model = getattr(model, "Model", model)
        old_use_gating = None
        if eval_disable_gating and hasattr(inner_model, "use_gating"):
            old_use_gating = bool(inner_model.use_gating)
            inner_model.use_gating = False

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    labels = self._prep_labels(batch_data["labels"]["M"].to(self.args.device))
                    text, audio_in, vision_in = self._pack_av_inputs(batch_data)

                    if eval_missing in ("a", "audio", "av"):
                        audio_in = self._zero_like_av(audio_in)
                    if eval_missing in ("v", "vision", "av"):
                        vision_in = self._zero_like_av(vision_in)

                    outs = model(text, audio_in, vision_in)
                    pred = outs["M"]
                    loss = self.criterion(pred, labels)
                    if self.branch_loss_weight > 0:
                        for k in ("y_t", "y_ta", "y_tv"):
                            if k in outs:
                                loss = loss + self.branch_loss_weight * self.criterion(outs[k], labels)

                    eval_loss += loss.item()
                    y_pred.append(pred.detach().cpu())
                    y_true.append(labels.detach().cpu())

        if old_use_gating is not None:
            inner_model.use_gating = old_use_gating

        eval_loss = eval_loss / len(dataloader)
        pred_all, true_all = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred_all, true_all)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")
        return eval_results

