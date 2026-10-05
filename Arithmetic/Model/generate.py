"""Generate arithmetic problem solutions from a trained GPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from model import GPTConfig, GPTLanguageModel

EOF_MARKER = "<EOF>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/gpt.pt"),
        help="checkpoint saved by train.py (default: checkpoints/gpt.pt)",
    )
    parser.add_argument("--prompt", default="5 + 6 = ?\n\n", help="text to continue")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k filtering")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--keep-eof",
        action="store_true",
        help="keep text generated after the first <EOF> marker",
    )
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    if args.temperature <= 0:
        parser.error("temperature must be positive")
    if args.top_k < 0:
        parser.error("top-k cannot be negative")
    return args


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[GPTLanguageModel, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = GPTLanguageModel(GPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["vocabulary"]


@torch.no_grad()
def generate_text(
    model: GPTLanguageModel,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        context = token_ids[:, -model.config.block_size :]
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_values[:, [-1]]] = -float("inf")
        probabilities = F.softmax(logits, dim=-1)
        # next_token = torch.multinomial(probabilities, num_samples=1)
        next_token = torch.argmax(probabilities, dim=-1, keepdim=True)
        token_ids = torch.cat((token_ids, next_token), dim=1)
    return token_ids


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    model, vocabulary = load_model(args.checkpoint, device)
    token_to_id = {character: index for index, character in enumerate(vocabulary)}

    unknown = sorted(set(args.prompt) - token_to_id.keys())
    if unknown:
        raise ValueError(f"Prompt contains characters unseen during training: {unknown}")
    prompt_ids = torch.tensor(
        [[token_to_id[character] for character in args.prompt]],
        dtype=torch.long,
        device=device,
    )

    output_ids = generate_text(
        model,
        prompt_ids,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
    )
    text = "".join(vocabulary[index] for index in output_ids[0].tolist())
    if not args.keep_eof and EOF_MARKER in text:
        text = text[: text.index(EOF_MARKER) + len(EOF_MARKER)]
    print(text)


if __name__ == "__main__":
    main()
