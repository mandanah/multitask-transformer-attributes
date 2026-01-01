from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, f1_score


def evaluate(model, loader, device: str):
    """Evaluate the model on the validation dataset."""
    model.eval()
    total_loss = 0
    n = 0.0
    sleeve_true, sleeve_pred = [], []
    pocket_true, pocket_pred = [], []
    buttons_true, buttons_pred = [], []

    for batch in loader:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        sleeve = batch.sleeve.to(device)
        pocket = batch.pocket.to(device)
        buttons = batch.buttons.to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sleeve=sleeve,
            pocket=pocket,
            buttons=buttons,
        )
        total_loss += out["total_loss"].item()
        n += 1

        sleeve_pred = out["sleeve_logits"].argmax(dim=-1).cpu().tolist()
        pocket_pred = out["pocket_logits"].argmax(dim=-1).cpu().tolist()
        buttons_pred = (torch.sigmoid(out["buttons_logit"]) > 0.5).long().cpu().tolist()

        sleeve_true.extend(sleeve.cpu().tolist())
        pocket_true.extend(pocket.cpu().tolist())
        buttons_true.extend(buttons.cpu().tolist())

    return {
        "val_loss": total_loss / max(n, 1),
        "sleeve_acc": accuracy_score(sleeve_true, sleeve_pred),
        "pocket_acc": accuracy_score(pocket_true, pocket_pred),
        "buttons_acc": accuracy_score(buttons_true, buttons_pred),
        "sleeve_f1": f1_score(sleeve_true, sleeve_pred, average="macro"),
        "pocket_f1": f1_score(pocket_true, pocket_pred, average="macro"),
        "buttons_f1": f1_score(buttons_true, buttons_pred),
    }
