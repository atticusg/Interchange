import torch
import torch.nn as nn

from typing import Dict, Union, Tuple
from antra.location import location_to_str

class Probe(nn.Module):
    def __init__(self,
                 high_node: str,
                 low_node: str,
                 low_loc: Union[int, Tuple, slice],
                 is_control: bool,
                 probe_output_classes: int,
                 probe_input_dim: int,
                 probe_max_rank: int,
                 probe_dropout: float):
        super(Probe, self).__init__()
        self.high_node = high_node
        self.low_node = low_node
        self.low_loc = low_loc
        self.is_control = is_control

        low_loc_str = location_to_str(low_loc).strip("[]")
        low_loc_str = low_loc_str.replace("::","x").replace(":",".").replace(",","_")
        self.name = f"{high_node}-{low_node}-{low_loc_str}{'-ctrl' if is_control else ''}"

        self.probe_output_classes = probe_output_classes
        self.probe_input_dim = probe_input_dim
        self.probe_max_rank = probe_max_rank
        self.probe_dropout = probe_dropout

        self.linear1 = nn.Linear(probe_input_dim, probe_max_rank)
        self.linear2 = nn.Linear(probe_max_rank, probe_output_classes)

        self.dropout = nn.Dropout(probe_dropout)

    def config(self):
        return {
            "high_node": self.high_node,
            "low_node": self.low_node,
            "low_loc": self.low_loc,
            "is_control": self.is_control,
            "probe_output_classes": self.probe_output_classes,
            "probe_input_dim": self.probe_input_dim,
            "probe_max_rank": self.probe_max_rank,
            "probe_dropout": self.probe_dropout
        }

    @classmethod
    def from_checkpoint(cls, save_path, opts=None):
        checkpoint = torch.load(save_path)
        assert 'model_config' in checkpoint
        model_config = checkpoint['model_config']
        if opts:
            model_config.update(opts)
        model = cls(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def forward(self, batch):
        batch = self.dropout(batch)
        batch = self.linear1(batch)
        logits = self.linear2(batch)
        return logits

