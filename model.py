import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from utils.build_data import SLEEVE_VOCAB, POCKET_VOCAB


class MultiTaskAttributeModel(nn.Module):
    """Multi-task model for predicting product attributes."""

    def __init__(self, base_model_name: str, dropout=0.1):
        super().__init__()
        self.model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.model.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.sleeve_classifier = nn.Linear(hidden_size, len(SLEEVE_VOCAB))  #
        self.pocket_classifier = nn.Linear(hidden_size, len(POCKET_VOCAB))  #
        self.buttons_classifier = nn.Linear(hidden_size, 1)  # binary

        self.buttons_loss = nn.BCEWithLogitsLoss()
        self.pocket_loss = nn.CrossEntropyLoss()
        self.sleeve_loss = nn.CrossEntropyLoss()

    def forward(
        self, input_ids, attention_mask, sleeve=None, pocket=None, buttons=None
    ) -> dict:  # specify the types
        """Forward pass of the model."""
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)

        sleeve_logits = self.sleeve_classifier(cls)
        pocket_logits = self.pocket_classifier(cls)
        buttons_logit = self.buttons_classifier(cls).squeeze(-1)

        total_loss = 0
        if sleeve is not None:
            sleeve_loss = self.sleeve_loss(sleeve_logits, sleeve)
            total_loss += sleeve_loss
        if pocket is not None:
            pocket_loss = self.pocket_loss(pocket_logits, pocket)
            total_loss += pocket_loss
        if buttons is not None:
            buttons_loss = self.buttons_loss(buttons_logit, buttons)
            total_loss += buttons_loss

        return {
            "total_loss": total_loss,
            "sleeve_logits": sleeve_logits,
            "pocket_logits": pocket_logits,
            "buttons_logit": buttons_logit,
        }
