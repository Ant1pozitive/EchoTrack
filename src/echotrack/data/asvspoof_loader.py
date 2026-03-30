"""
Hugging Face loader for ASVspoof 5 dataset.
Provides train/dev/eval splits with labels (bonafide/spoof).
"""

from datasets import load_dataset
import numpy as np
from typing import Dict, Iterator, Tuple


def load_asvspoof5(split: str = "train", streaming: bool = True) -> Iterator[Dict]:
    """
    Load ASVspoof 5 dataset from Hugging Face.

    Args:
        split: One of 'train', 'dev', 'eval' (check actual protocol names in dataset)
        streaming: Use streaming mode for large dataset

    Returns:
        Iterator over dataset examples
    """
    # ASVspoof 5 is hosted at jungjee/asvspoof5
    dataset = load_dataset(
        "jungjee/asvspoof5",
        split=split,
        streaming=streaming,
        trust_remote_code=True
    )

    for example in dataset:
        # Example structure usually contains: audio (dict with array and sampling_rate), label, etc.
        yield {
            "audio": example["audio"]["array"],        # raw waveform
            "sampling_rate": example["audio"]["sampling_rate"],
            "label": example.get("label", 0),          # 0 = bonafide, 1 = spoof
            "attack_type": example.get("attack_type", None),
            "speaker_id": example.get("speaker_id", None),
        }


def get_label_name(label: int) -> str:
    """Convert numeric label to human-readable string."""
    return "bonafide" if label == 0 else "spoof"
