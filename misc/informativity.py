import torch
import intervention
from intervention.utils import serialize
from compgraphs import MQNLI_Logic_CompGraph
from torch.utils.data import DataLoader, Subset, IterableDataset


class InterventionDataset(IterableDataset):
    def __init__(self, model):
        super(InterventionDataset, self).__init__()
        self.model = model
        self.examples = []

    @property
    def num_examples(self):
        return len(self.examples)

    def add_examples(self, examples):
        self.examples.extend(examples)

def main(num_inputs, batch_size):
    data_path = "../data/mqnli/preprocessed/bert-easy.pt"
    data = torch.load(data_path)
    high_model = MQNLI_Logic_CompGraph(data)
    subset = Subset(data.dev, list(range(num_inputs)))
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    ivd = InterventionDataset(high_model)
    for i, input_tuple in enumerate(dataloader):
        high_input_tensor = input_tuple[-2]
        high_base_key = [serialize(x) for x in high_input_tensor]
        high_input = intervention.GraphInput.batched(
            {"input": high_input_tensor.T}, high_base_key)
        high_model.compute(high_input)
        ivd.add_examples(high_input_tensor)



if __name__ == "__main__":
    with torch.no_grad():
        main(num_inputs=200, batch_size=100)
