import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import random
from dataset import ProductTextDataset
from model import MultiTaskAttributeModel
from torch.utils.data import DataLoader
from utils.collate import collate_fn
from torch.optim import AdamW
from eval import evaluate
from utils.build_data import build_toy_data


def train(base_model_name: str, rows, epochs: int):
    """Train the multi-task attribute model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    pad_id = tokenizer.pad_token_id

    random.shuffle(rows)
    split_idx = int(0.8 * len(rows))
    train_rows, val_rows = rows[:split_idx], rows[split_idx:]

    model = MultiTaskAttributeModel(base_model_name).to(device)

    train_dataset = ProductTextDataset(train_rows, tokenizer)
    val_dataset = ProductTextDataset(val_rows, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_id),
    )

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            sleeve = batch.sleeve.to(device)
            pocket = batch.pocket.to(device)
            buttons = batch.buttons.to(device)

            optimizer.zero_grad()
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sleeve=sleeve,
                pocket=pocket,
                buttons=buttons,
            )
            loss = out["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()

        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: {metrics}")
        print(f"  Train Loss: {running / len(train_loader)}")
        print(
            f"| sleeve_acc={metrics['sleeve_acc']} pocket_acc={metrics['pocket_acc']} buttons_acc={metrics['buttons_acc']}"
        )
        save_dir = "./phase1_multitask_model"
        tokenizer.save_pretrained(save_dir)
        torch.save(model.state_dict(), f"{save_dir}/model.pt")
        print(f"Saved tokenizer + model state_dict to {save_dir}")


def main():
    base_model_name = "distilbert-base-uncased"
    rows = build_toy_data()
    train(base_model_name, rows, epochs=3)


if __name__ == "__main__":
    main()
