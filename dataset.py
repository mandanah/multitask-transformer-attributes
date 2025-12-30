from torch.utils.data import Dataset
from utils.build_data import SLEEVE_VOCAB, POCKET_VOCAB


class ProductTextDataset(Dataset):
    """Custom Dataset for product text data."""

    def __init__(self, rows: list[dict], tokenizer, max_length: int = 128):
        super().__init__()
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]

        text = item["text"][: self.max_length]
        sleeve_length = SLEEVE_VOCAB[item["sleeve_length"]]
        pocket_type = POCKET_VOCAB[item["pocket_type"]]
        buttons = int(item["has_buttons"])

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "sleeve": sleeve_length,
            "pocket": pocket_type,
            "buttons": buttons,
        }
