"""
AMIO -- All Model in One (minimal)

This minimal version only supports the CoReGate model for paper release.
"""

import torch.nn as nn

from .singleTask import CoReGate


class AMIO(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.MODEL_MAP = {
            "coregate": CoReGate,
        }
        model_cls = self.MODEL_MAP[args["model_name"]]
        self.Model = model_cls(args)

    def forward(self, text_x, audio_x, video_x, *args, **kwargs):
        return self.Model(text_x, audio_x, video_x, *args, **kwargs)
