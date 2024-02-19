import torch
from typing import Dict

from interface import OODDetector

class EnergyOODDetector(OODDetector):

    def setup(self, args, id_train_model_outputs):
        pass

    def infer(self, model_outputs):
        assert "logits" in model_outputs
        logits = model_outputs["logits"]
        # without temperature scaling, T=1
        return torch.logsumexp(logits, dim=1)