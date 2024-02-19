from interface import OODDetector

class MaxLogitOODDetector(OODDetector):

    def setup(self, args, train_model_outputs):
        pass

    def infer(self, model_outputs):
        logits = model_outputs['logits']
        return logits.max(dim=1)[0]