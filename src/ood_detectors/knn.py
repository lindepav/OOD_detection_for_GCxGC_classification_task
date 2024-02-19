import torch
from typing import Dict
from interface import OODDetector
from ood_detectors.detectors_utils import knn_score
import numpy as np

class KNNOODDetector(OODDetector):
    def setup(self, args, train_model_outputs):
        feas_train = train_model_outputs['feas']
        if isinstance(feas_train, np.ndarray):
            feas_train = torch.from_numpy(feas_train)
        try:
            self.knn_k = args.detector['knn_k']
        except:
            self.knn_k = 10

        self.feas_train = feas_train

    def infer(self, model_outputs):
        feas = model_outputs['feas']
        if isinstance(feas, np.ndarray):
            feas = torch.from_numpy(feas)
        scores = knn_score(self.feas_train, feas, k=self.knn_k, min=True)
        scores = torch.from_numpy(scores).to(feas.device)
        return scores
