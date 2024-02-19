from interface import OODDetector
import torch
from ood_detectors.detectors_utils import Mahalanobis
import numpy as np


class SSDOODDetector(OODDetector):

    def setup(self, args, train_model_outputs):
        feas_train = train_model_outputs['feas']

        # self.ssd = Mahalanobis(num_clusters=np.unique(train_model_outputs['labels_true']).size)
        self.ssd = Mahalanobis(num_clusters=1)
        self.ssd.fit(feas_train)

    def infer(self, model_outputs):
        feas = model_outputs['feas']
        if isinstance(feas, np.ndarray):
            feas = torch.from_numpy(feas)
        device = feas.device
        scores = torch.from_numpy(self.ssd.score(feas, return_distance=False)).to(device)
        return scores
        # return torch.from_numpy(self.ssd.score(feas)).to(device)