"""Download the ViWiki dataset from HuggingFace to local disk."""

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ViWiki dataset")
    parser.add_argument(
        "--config",
        default="cleaned",
        choices=["cleaned", "raw"],
        help="Dataset config (default: cleaned)",
    )
    parser.add_argument(
        "--output",
        default="data/viwiki-cleaned",
        help="Output directory (default: data/viwiki-cleaned)",
    )
    parser.add_argument(
        "--dataset-id",
        default="Keithsel/viwiki-20260523",
        help="HuggingFace dataset ID",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.dataset_id} (config={args.config})...")
    ds = load_dataset(args.dataset_id, args.config, split="train")
    print(f"Downloaded {len(ds)} articles")

    ds.save_to_disk(str(output_path))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
