# Multi-Task Transformer for Product Attributes

A PyTorch-based multi-task learning system for predicting clothing product attributes from text descriptions using transformer models.

## Overview

This project implements a multi-task neural network that jointly learns to predict three clothing attributes from product descriptions:

- **Sleeve Length**: Multi-class classification (sleeveless, short, long)
- **Pocket Type**: Multi-class classification (none, patch, kangaroo)
- **Buttons**: Binary classification (present/absent)

The model uses a pre-trained DistilBERT transformer as a feature extractor with separate classification heads for each task, enabling efficient joint learning across related prediction tasks.

## Project Structure

```
.
├── model.py                    # Multi-task model architecture
├── dataset.py                  # PyTorch dataset implementation
├── eval.py                     # Evaluation metrics and functions
├── main.py                     # Training loop and experiment runner
├── config.py                   # Configuration settings
├── schema.py                   # Data schemas (Batch dataclass)
├── utils/
│   ├── build_data.py          # Data preprocessing and vocabularies
│   └── collate.py             # Batch collation for DataLoader
├── phase1_multitask_model/    # Saved model checkpoint
│   ├── model.pt               # Trained model weights
│   └── tokenizer files        # BERT tokenizer configuration
├── pyproject.toml             # Project dependencies
└── uv.lock                    # Dependency lock file
```

## How It Works

### 1. Model Architecture (`model.py`)

The `MultiTaskAttributeModel` is built on top of a pre-trained transformer:

```
Input Text → DistilBERT → [CLS] Token → Dropout → 3 Classification Heads
                                            ├─→ Sleeve Classifier (3 classes)
                                            ├─→ Pocket Classifier (3 classes)
                                            └─→ Buttons Classifier (binary)
```

- **Base Model**: DistilBERT (`distilbert-base-uncased`)
- **Feature Extraction**: Uses [CLS] token from the last hidden state
- **Loss Functions**:
  - Sleeve & Pocket: CrossEntropyLoss
  - Buttons: BCEWithLogitsLoss
- **Total Loss**: Sum of all task-specific losses

### 2. Dataset Pipeline (`dataset.py`, `utils/`)

**Data Flow:**
1. Raw text → `build_toy_data()` generates sample data
2. Text tokenization → `ProductTextDataset` processes with BERT tokenizer
3. Batch creation → `collate_fn()` pads sequences and creates tensors
4. Training → DataLoader feeds batches to model

**Label Encoding:**
```python
SLEEVE_VOCAB = {"sleeveless": 0, "short": 1, "long": 2}
POCKET_VOCAB = {"none": 0, "patch": 1, "kangaroo": 2}
```

### 3. Training Loop (`main.py`)

**Training Configuration:**
- **Optimizer**: AdamW (lr=5e-5, weight_decay=0.01)
- **Scheduler**: Linear warmup (10% of total steps)
- **Batch Size**: 4
- **Epochs**: 3
- **Train/Val Split**: 80/20
- **Gradient Clipping**: max_norm=1.0

**Training Steps:**
1. Load pre-trained DistilBERT and tokenizer
2. Split data into train/validation sets
3. For each epoch:
   - Forward pass through model
   - Compute combined loss
   - Backward pass and optimizer step
   - Evaluate on validation set
   - Save model checkpoint

### 4. Evaluation (`eval.py`)

Computes metrics for each task:
- **Accuracy**: Per-task classification accuracy
- **F1 Score**: Macro-averaged F1 for sleeve/pocket, binary F1 for buttons
- **Validation Loss**: Combined loss across all tasks

## Installation

This project uses `uv` for dependency management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

**Dependencies:**
- Python ≥ 3.11
- PyTorch ≥ 2.9.1
- Transformers ≥ 4.57.3
- scikit-learn ≥ 1.8.0

## Usage

### Training

Run the training script:

```bash
python main.py
```

This will:
1. Generate toy training data (6 samples)
2. Train for 3 epochs
3. Save model and tokenizer to `./phase1_multitask_model/`

### Using the Trained Model

```python
import torch
from transformers import AutoTokenizer
from model import MultiTaskAttributeModel
from utils.build_data import INV_SLEEVE, INV_POCKET

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiTaskAttributeModel("distilbert-base-uncased").to(device)
model.load_state_dict(torch.load("phase1_multitask_model/model.pt"))
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("phase1_multitask_model")

# Make prediction
text = "women hoodie long sleeve with kangaroo pocket"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    
sleeve_pred = outputs["sleeve_logits"].argmax(dim=-1).item()
pocket_pred = outputs["pocket_logits"].argmax(dim=-1).item()
buttons_pred = (torch.sigmoid(outputs["buttons_logit"]) > 0.5).item()

print(f"Sleeve: {INV_SLEEVE[sleeve_pred]}")
print(f"Pocket: {INV_POCKET[pocket_pred]}")
print(f"Buttons: {buttons_pred}")
```

## Key Features

### Multi-Task Learning Benefits
- **Shared Representations**: Common feature extraction benefits all tasks
- **Improved Generalization**: Related tasks help regularize each other
- **Efficiency**: Single model handles multiple predictions

### Design Patterns
- **Dataclass Schema** (`schema.py`): Type-safe batch representation
- **Custom Collation** (`collate.py`): Handles variable-length sequences
- **Modular Architecture**: Separate concerns (data, model, training, eval)

## Data Format

Input data should be a list of dictionaries:

```python
{
    "text": "women hoodie long sleeve with kangaroo pocket",
    "sleeve_length": "long",        # "sleeveless" | "short" | "long"
    "pocket_type": "kangaroo",      # "none" | "patch" | "kangaroo"
    "has_buttons": 0                # 0 or 1
}
```

## Extending the Project

### Adding New Attributes
1. Define vocabulary in `utils/build_data.py`
2. Add classifier head in `model.py`
3. Update loss computation in `forward()`
4. Modify `Batch` schema in `schema.py`
5. Update `collate_fn` and `ProductTextDataset`

### Using Real Data
Replace `build_toy_data()` in `utils/build_data.py` with your data loading logic:

```python
def load_real_data(file_path: str) -> list[dict]:
    # Load from CSV, JSON, or database
    # Return list of dicts with required fields
    pass
```

Then update `main.py`:
```python
rows = load_real_data("path/to/data.csv")
```

## Model Output

The model returns a dictionary with:
- `total_loss`: Combined loss for backpropagation
- `sleeve_logits`: Raw logits for sleeve classification (shape: [batch_size, 3])
- `pocket_logits`: Raw logits for pocket classification (shape: [batch_size, 3])
- `buttons_logit`: Raw logit for button prediction (shape: [batch_size])

## Performance Notes

- **Current Setup**: Toy data with 6 samples (proof of concept)
- **Production Use**: Requires substantial training data (1000+ samples per class)
- **GPU Recommended**: Training is significantly faster with CUDA
- **Inference Speed**: ~10-50ms per sample on CPU, <5ms on GPU

## Future Improvements

- [ ] Add data augmentation
- [ ] Implement early stopping
- [ ] Add learning rate finder
- [ ] Support for additional base models (RoBERTa, ALBERT)
- [ ] Implement class weights for imbalanced data
- [ ] Add confidence thresholds for predictions
- [ ] Create inference API endpoint

## License

MIT

## Contributing

Contributions are welcome! Please ensure code follows the existing structure and includes appropriate type hints.
