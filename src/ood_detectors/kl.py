import torch
from typing import Dict
from interface import OODDetector
import torch.nn.functional as F


class KLOODDetector(OODDetector):
    def setup(self, args, train_model_outputs):
        pass

    def infer(self, model_outputs: Dict):
        logits = model_outputs['logits']
        probs = torch.nn.functional.softmax(logits, dim=1)
        Q = (torch.ones_like(probs) / probs.shape[-1])
        return F.cross_entropy(logits, Q, reduction='none')