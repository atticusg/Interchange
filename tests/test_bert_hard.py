import torch
from torch.utils.data import DataLoader
from trainer import load_model, evaluate_and_predict
from modeling.pretrained_bert import PretrainedBertModule
from tqdm import tqdm

model_save_path = "data/training/test/bert-test.pt"
hard_data_path = "data/mqnli/preprocessed/bert-hard.pt"
easy_data_path = "data/mqnli/preprocessed/bert-easy.pt"

def test_full_sentence_labels():
    device = torch.device("cuda")
    model, _ = load_model(PretrainedBertModule, model_save_path)
    model = model.to(device)
    model.eval()

    hard_data = torch.load(hard_data_path)
    dataloader = DataLoader(hard_data.dev, batch_size=64, shuffle=False)

    total_preds = 0
    correct_preds = 0
    invalid_preds = 0

    with torch.no_grad():
        for input_tuple in tqdm(dataloader):
            input_tuple = [x.to(device) for x in input_tuple]
            labels = input_tuple[-1]

            logits = model(input_tuple)
            pred = torch.argmax(logits, dim=1)

            invalid_preds += torch.sum(pred >= 3).item()
            correct_in_batch = torch.sum(torch.eq(pred, labels)).item()
            total_preds += labels.shape[0]
            correct_preds += correct_in_batch

    print(f"hard dev accuracy {correct_preds}/{total_preds}={correct_preds/total_preds:.2%}")
    print(f"invalid rate {invalid_preds}/{total_preds}={invalid_preds/total_preds:.2%}")

    print("loading easy data")
    easy_data = torch.load(easy_data_path)

    easy_correct, easy_total = evaluate_and_predict(easy_data.dev, model)
    print(f"easy dev accuracy {easy_correct}/{easy_total}={easy_correct / easy_total:.2%}")