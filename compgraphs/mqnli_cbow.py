import torch

from intervention import ComputationGraph
from intervention import GraphNode

class MQNLI_CBOW_CompGraph(ComputationGraph):
    def __init__(self, cbow_model):
        if cbow_model.task != "mqnli":
            raise ValueError("The CBOW model must be for MQNLI!")
        self.model = cbow_model

        @GraphNode()
        def input(x):
            # (18,) or (18, batch_size)
            if len(x.shape) == 1:
                x = x.unsqueeze(1)
            assert len(x.shape) == 2 and x.shape[0] == 18, \
                f"x.shape is {x.shape}"
            return self.model.embedding((x,))

        @GraphNode(input)
        def premise_emb(emb):
            return self.model.premise_mean(emb)

        @GraphNode(input)
        def hypothesis_emb(emb):
            return self.model.hypothesis_mean(emb)

        @GraphNode(premise_emb, hypothesis_emb)
        def concat(p, h):
            return self.model.concat(p, h)

        @GraphNode(concat)
        def ffnn1(x):
            return self.model.feed_forward1(x)

        @GraphNode(ffnn1)
        def activation1(x):
            return self.model.activation1(x)

        @GraphNode(activation1)
        def ffnn2(x):
            return self.model.feed_forward2(x)

        @GraphNode(ffnn2)
        def activation2(x):
            return self.model.activation2(x)

        @GraphNode(activation2)
        def logits(x):
            h_dim = self.model.hidden_dim * 2
            assert (len(x.shape) == 2 and x.shape[1] == h_dim)

            return self.model.logits(x)

        @GraphNode(logits)
        def root(x):
            return torch.argmax(x, dim=1)

        super(MQNLI_CBOW_CompGraph, self).__init__(root)
