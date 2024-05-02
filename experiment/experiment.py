'''
Class hosting the "experiment" of the specified model and RCT, computes and outputs a ground truth TTE and its estimate using an estimator.
'''
class Experiment:
    def __init__(self, model, rct) -> None:
        self.model = model
        self.rct = rct

    def run(self, numIterations):
        for _ in range(numIterations):
            tte = self.computeTTE()
            estimate = self.computeEstimate()
        # TODO

    def computeTTE(self):
        '''
        Computes the TTE from the POM of the model.
        '''
        pass

    def computeEstimate(self):
        '''
        Computes the estimate of the TTE using the estimator.
        '''
        pass