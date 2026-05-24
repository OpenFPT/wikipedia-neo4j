"""Download UIT-ViQuAD2.0 and export to local JSONL cache."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.viquad_adapter import export_eval_jsonl


if __name__ == "__main__":
    print("Downloading and caching ViQuAD2.0 validation split...")
    path = export_eval_jsonl("validation")
    print(f"Exported to {path}")

    print("Downloading and caching ViQuAD2.0 test split...")
    path = export_eval_jsonl("test")
    print(f"Exported to {path}")
