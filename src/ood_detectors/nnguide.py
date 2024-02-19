from interface import OODDetector
import torch
from ood_detectors.detectors_utils import knn_score
import numpy as np


class NNGuideOODDetector(OODDetector):

    def setup(self, args, train_model_outputs):
        logits_train = train_model_outputs['logits']
        feas_train = train_model_outputs['feas']
        if isinstance(feas_train, np.ndarray):
            feas_train = torch.from_numpy(feas_train)
        try:
            self.knn_k = args.detector_parameters['knn_k']
        except:
            self.knn_k = 10

        confs_train = torch.logsumexp(logits_train, dim=1)   # energy function
        self.scaled_feas_train = feas_train * confs_train[:, None]

    def infer(self, model_outputs):
        feas = model_outputs['feas']
        logits = model_outputs['logits']

        confs = torch.logsumexp(logits, dim=1)              # energy function
        guidances = knn_score(self.scaled_feas_train, feas, k=self.knn_k)
        scores = torch.from_numpy(guidances).to(confs.device) * confs
        return scores