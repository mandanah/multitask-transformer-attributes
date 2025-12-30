import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from utils.build_data import SLEEVE_VOCAB, POCKET_VOCAB


class MultiTaskAttributeModel(nn.Module):
    def __init__(self, base_model_name: str, dropout=0.1):
        super().__init__()
        self.model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.model.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.sleeve_classifier = nn.Linear(hidden_size, len(SLEEVE_VOCAB))  #
        self.pocket_classifier = nn.Linear(hidden_size, len(POCKET_VOCAB))  #
        self.buttons_classifier = nn.Linear(hidden_size, 1)  # binary

    def forward(
        self, input_ids, attention_mask, sleeve=None, pocket=None, buttons=None
    ):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
