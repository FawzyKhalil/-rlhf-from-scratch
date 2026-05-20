"""
Prepare Anthropic HH-RLHF for reward model and SFT training.

For each row the dataset provides two full conversations:
  chosen  — prompt + preferred assistant response
  rejected — same prompt + dispreferred response

We extract the shared prompt by splitting on the last '\n\nAssistant:' occurrence,
then tokenise both (prompt+chosen_response) and (prompt+rejected_response) with
the GPT-2 tokeniser, truncating to max_length=512.

Output: HuggingFace DatasetDict saved to disk (Apache Arrow format):
  data/processed/{train,val,test}

Usage:
    python data/prepare_hh_rlhf.py [--output_dir data/processed]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, Dataset, load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm


ASSISTANT_MARKER = "\n\nAssistant:"


def split_prompt_response(text: str) -> tuple[str, str]:
    """Return (prompt_including_marker, response) by splitting at the last assistant turn."""
    idx = text.rfind(ASSISTANT_MARKER)
    if idx == -1:
        return text, ""
    prompt = text[: idx + len(ASSISTANT_MARKER)]
    response = text[idx + len(ASSISTANT_MARKER) :]
    return prompt, response


def tokenise(tokenizer: GPT2Tokenizer, prompt: str, response: str, max_length: int) -> dict:
    """Tokenise prompt+response as one sequence, right-truncated to max_length."""
    enc = tokenizer(
        prompt + response,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }


def process_split(
    hf_split,
    tokenizer: GPT2Tokenizer,
    max_length: int,
) -> list[dict]:
    records = []
    for ex in tqdm(hf_split, desc="tokenising"):
        prompt, chosen_response = split_prompt_response(ex["chosen"])
        _, rejected_response = split_prompt_response(ex["rejected"])

        if not chosen_response.strip() or not rejected_response.strip():
            continue

        chosen_enc = tokenise(tokenizer, prompt, chosen_response, max_length)
        rejected_enc = tokenise(tokenizer, prompt, rejected_response, max_length)

        records.append(
            {
                "prompt": prompt,
                "chosen_response": chosen_response,
                "rejected_response": rejected_response,
                "chosen_input_ids": chosen_enc["input_ids"],
                "chosen_attention_mask": chosen_enc["attention_mask"],
                "rejected_input_ids": rejected_enc["input_ids"],
                "rejected_attention_mask": rejected_enc["attention_mask"],
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HH-RLHF dataset")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--subset",
        default="harmless-base",
        choices=["harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"],
    )
    parser.add_argument("--val_fraction", type=float, default=0.05)
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Anthropic/hh-rlhf ({args.subset})…")
    raw = load_dataset("Anthropic/hh-rlhf", data_dir=args.subset)

    train_records = process_split(raw["train"], tokenizer, args.max_length)
    test_records = process_split(raw["test"], tokenizer, args.max_length)

    n_val = max(1, int(len(train_records) * args.val_fraction))
    val_records = train_records[-n_val:]
    train_records = train_records[:-n_val]

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_records),
            "val": Dataset.from_list(val_records),
            "test": Dataset.from_list(test_records),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))

    print(f"\nSaved to {output_dir}")
    print(f"  train: {len(train_records):,} pairs")
    print(f"  val:   {len(val_records):,} pairs")
    print(f"  test:  {len(test_records):,} pairs")


if __name__ == "__main__":
    main()
