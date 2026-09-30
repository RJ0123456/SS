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
    parser.add_argument("--block-size", type=int, default=256)
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


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(data) - block_size, (batch_size,))
    inputs = torch.stack([data[start : start + block_size] for start in starts])
    targets = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPTLanguageModel,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
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
    vocabulary = sorted(set(text))
    token_to_id = {character: index for index, character in enumerate(vocabulary)}
    encoded = torch.tensor([token_to_id[character] for character in text], dtype=torch.long)

    validation_size = int(len(encoded) * args.validation_fraction)
    training_size = len(encoded) - validation_size
    if min(training_size, validation_size) <= args.block_size:
        raise ValueError(
            "The training and validation splits must each contain more characters "
            "than block-size; use more data or a smaller --block-size."
        )
    train_data = encoded[:training_size]
    validation_data = encoded[training_size:]

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

    print(f"Training on {device}; corpus={len(encoded):,} characters; vocabulary={len(vocabulary)}")
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