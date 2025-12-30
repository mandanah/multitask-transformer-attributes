def build_toy_data() -> list[dict]:
    # Tiny toy data: just to prove end-to-end wiring works.
    # Replace with real data for meaningful results.
    rows = [
        {
            "text": "women hoodie long sleeve with kangaroo pocket",
            "sleeve_length": "long",
            "pocket_type": "kangaroo",
            "has_buttons": 0,
        },
        {
            "text": "men tee short sleeve with patch pocket",
            "sleeve_length": "short",
            "pocket_type": "patch",
            "has_buttons": 0,
        },
        {
            "text": "button down shirt long sleeve",
            "sleeve_length": "long",
            "pocket_type": "none",
            "has_buttons": 1,
        },
        {
            "text": "tank top sleeveless",
            "sleeve_length": "sleeveless",
            "pocket_type": "none",
            "has_buttons": 0,
        },
        {
            "text": "hoodie short sleeve kangaroo pocket",
            "sleeve_length": "short",
            "pocket_type": "kangaroo",
            "has_buttons": 0,
        },
        {
            "text": "utility shirt long sleeve patch pocket buttons",
            "sleeve_length": "long",
            "pocket_type": "patch",
            "has_buttons": 1,
        },
    ]
    return rows


# labels
SLEEVE_VOCAB = {"sleeveless": 0, "short": 1, "long": 2}
POCKET_VOCAB = {"none": 0, "patch": 1, "kangaroo": 2}

INV_SLEEVE = {v: k for k, v in SLEEVE_VOCAB.items()}
INV_POCKET = {v: k for k, v in POCKET_VOCAB.items()}
