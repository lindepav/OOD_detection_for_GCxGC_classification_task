import torch
import torch.nn.functional as F
from interface import OODDetector


class MSPOODDetector(OODDetector):

    def setup(self, args, id_train_model_outputs):
        pass

    def infer(self, model_outputs):
        logits = model_outputs['logits']
        # TODO: check if this is the right way to normalize the logits
        # logits /= logits.norm(dim=1, keepdim=True)
        scores = F.softmax(logits, dim=1).max(dim=1)[0]
        return scores