from modeling.pretrained_bert import PretrainedBertModule
from trainer import load_model
import torch
from torch.utils.data import DataLoader

model_save_path = "data/models/bert-easy-best.pt"
data_path = "data/preprocessed/bert-easy.pt"
model, _ = load_model(PretrainedBertModule, model_save_path)
data = torch.load(data_path)
dataloader = DataLoader(data.dev, batch_size=1, shuffle=False)

print("num bert layers:", len(model.bert.encoder.layer))

for input_tuple in dataloader:
    input_ids = input_tuple[0]
    token_type_ids = input_tuple[1]
    attention_mask = input_tuple[2]
    unshifted_ids = input_tuple[3]
    label = input_tuple[4]
    input_tuple = [x.to(model.device) for x in input_tuple]
    break


print(input_tuple)
model(input_tuple)
