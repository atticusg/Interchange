class InterchangeExperiment(object):
    def __init__(self, trained_model, interventions, map):
        for intervention in interventions:
            trained_model.run_counterfactual(intervention)
