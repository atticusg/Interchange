import torch
from modeling.lstm import LSTMModule
from modeling.pretrained_bert import PretrainedBertModule

if __name__ == "__main__":
    load_path = "../data/models/bert-easy-best.pt"
    save_path = "../data/models/bert-baseline.pt"
    checkpoint = torch.load(load_path, map_location=torch.device("cpu"))

    assert 'model_config' in checkpoint
    model_config = checkpoint["model_config"]
    model = PretrainedBertModule(**model_config)

    baseline_model_checkpoint = {
        'model_name': "PretrainedBertModule",
        'epoch': None,
        'step': None,
        'duration': None,
        'model_state_dict': model.state_dict(),
        'loss': None,
        'best_def_acc': None,
        'train_config': checkpoint['train_config'],
        'model_config': model_config
    }

    torch.save(baseline_model_checkpoint, save_path)

    # model_config = checkpoint['model_config']
    # model = LSTMModule(**model_config)
    #
    # baseline_model_checkpoint = {
    #     'model_name': "LSTMModule",
    #     'epoch': None,
    #     'step': None,
    #     'duration': None,
    #     'model_state_dict': model.state_dict(),
    #     'loss': None,
    #     'best_def_acc': None,
    #     'train_config': checkpoint['train_config'],
    #     'model_config': model_config
    # }
    #
    # torch.save(baseline_model_checkpoint, save_path)
