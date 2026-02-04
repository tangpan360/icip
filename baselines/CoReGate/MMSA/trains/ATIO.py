"""
ATIO -- All Trains in One (minimal)

This minimal version only supports the CoReGate trainer.
"""

from .singleTask import CoReGate

__all__ = ["ATIO"]


class ATIO:
    def __init__(self):
        self.TRAIN_MAP = {
            "coregate": CoReGate,
        }

    def getTrain(self, args):
        return self.TRAIN_MAP[args["model_name"]](args)
