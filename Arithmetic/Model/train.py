"""Train the character-level arithmetic GPT model."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from model import GPTConfig, GPTLanguageModel, config_to_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Arithmetic/Data/generated"))
    parser.add_argument("--output", type=Path, default=Path("Arithmetic/Model/checkpoints/gpt.pt"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch-size must be positive")
    if args.block_size < 1 or args.eval_interval < 1 or args.eval_iters < 1:
        parser.error("block-size, eval-interval, and eval-iters must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("validation-fraction must be between 0 and 1")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning-rate must be positive and weight-decay cannot be negative")
    return args


def load_corpus(data_dir: Path) -> str:
    files = sorted(data_dir.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt training files found under {data_dir}")
    return "\n\n".join(path.read_text(encoding="utf-8") for path in files)


def split_examples(text: str) -> list[str]:
    """Split the corpus into complete examples, including their blank-line separator."""
    return [
        part.strip("\n") + "\n\n"
        for part in text.split("\n\n")
        if part.strip()
    ]


def get_batch(
    data: list[torch.Tensor],
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    examples = [data[index] for index in torch.randint(len(data), (batch_size,)).tolist()]
    sequence_lengths = [example.numel() - 1 for example in examples]
    sequence_length = max(sequence_lengths)
    if sequence_length > block_size:
        raise ValueError(
            f"A complete example needs {sequence_length} input positions, "
            f"exceeding block-size={block_size}; increase --block-size."
        )

    inputs = torch.zeros((batch_size, sequence_length), dtype=torch.long)
    targets = torch.full((batch_size, sequence_length), -100, dtype=torch.long)
    for row, (example, length) in enumerate(zip(examples, sequence_lengths)):
        inputs[row, :length] = example[:-1]
        targets[row, :length] = example[1:]
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPTLanguageModel,
    train_data: list[torch.Tensor],
    validation_data: list[torch.Tensor],
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    results: dict[str, float] = {}
    for split_name, data in (("train", train_data), ("validation", validation_data)):
        losses = []
        for _ in range(eval_iters):
            inputs, targets = get_batch(data, batch_size, block_size, device)
            _, loss = model(inputs, targets)
            losses.append(loss.item())
        results[split_name] = sum(losses) / len(losses)
    model.train()
    return results


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    text = load_corpus(args.data_dir)
    examples = split_examples(text)
    if len(examples) < 2:
        raise ValueError("At least two complete examples are required for training and validation.")

    vocabulary = sorted(set("".join(examples)))
    token_to_id = {character: index for index, character in enumerate(vocabulary)}
    encoded_examples = [
        torch.tensor([token_to_id[character] for character in example], dtype=torch.long)
        for example in examples
    ]

    longest_example = max(example.numel() - 1 for example in encoded_examples)
    if longest_example > args.block_size:
        raise ValueError(
            f"The longest complete example needs {longest_example} input positions, "
            f"exceeding block-size={args.block_size}; increase --block-size."
        )

    validation_size = max(1, int(len(encoded_examples) * args.validation_fraction))
    if validation_size >= len(encoded_examples):
        raise ValueError("The validation split leaves no complete examples for training.")
    training_size = len(encoded_examples) - validation_size
    train_data = encoded_examples[:training_size]
    validation_data = encoded_examples[training_size:]

    config = GPTConfig(
        vocab_size=len(vocabulary),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        block_size=args.block_size,
    )
    model = GPTLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    corpus_size = sum(example.numel() for example in encoded_examples)
    print(
        f"Training on {device}; examples={len(encoded_examples):,}; "
        f"corpus={corpus_size:,} characters; vocabulary={len(vocabulary)}"
    )
    for step in range(1, args.steps + 1):
        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            losses = estimate_loss(
                model,
                train_data,
                validation_data,
                args.batch_size,
                args.block_size,
                args.eval_iters,
                device,
            )
            print(
                f"step {step:>6}: train loss {losses['train']:.4f}, "
                f"validation loss {losses['validation']:.4f}"
            )

        inputs, targets = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config_to_dict(config),
            "vocabulary": vocabulary,
            "training_steps": args.steps,
        },
        args.output,
    )
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()