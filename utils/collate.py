from schema import Batch
import torch


def collate_fn(examples: list[dict], pad_token_id: int) -> Batch:
    """Custom collate function to batch examples."""
    max_length = max(len(ex["input_ids"]) for ex in examples)
    sleeve, pocket, buttons = [], [], []
    input_ids, attention_mask = [], []

    for example in examples:
        pad_len = max_length - len(example["input_ids"])

        ids = example["input_ids"]
        mask = example["attention_mask"]

        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_mask.append(mask + [0] * pad_len)

        sleeve.append(example["sleeve"])
        pocket.append(example["pocket"])
        buttons.append(example["buttons"])

    return Batch(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        sleeve=torch.tensor(sleeve, dtype=torch.long),
        pocket=torch.tensor(pocket, dtype=torch.long),
        buttons=torch.tensor(buttons, dtype=torch.float32),
    )
