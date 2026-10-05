from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from train import get_batch, load_corpus, split_examples


class SplitExamplesTests(unittest.TestCase):
    def test_splits_records_and_preserves_eof_and_separator(self) -> None:
        text = "first\n<EOF>\n\n\nsecond\n<EOF>\n\n"

        self.assertEqual(
            split_examples(text),
            ["first\n<EOF>\n\n", "second\n<EOF>\n\n"],
        )

    def test_ignores_empty_input(self) -> None:
        self.assertEqual(split_examples("\n\n"), [])


class LoadCorpusTests(unittest.TestCase):
    def test_loads_text_files_in_sorted_recursive_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "b.txt").write_text("second", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "a.txt").write_text("first", encoding="utf-8")
            (root / "ignored.md").write_text("ignored", encoding="utf-8")

            self.assertEqual(load_corpus(root), "second\n\nfirst")

    def test_raises_when_no_text_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(FileNotFoundError):
                load_corpus(Path(temporary_directory))


class GetBatchTests(unittest.TestCase):
    def test_samples_complete_examples_and_masks_padding_targets(self) -> None:
        data = [
            torch.tensor([10, 11, 12, 13]),
            torch.tensor([20, 21, 22]),
        ]
        with patch("train.torch.randint", return_value=torch.tensor([0, 1])):
            inputs, targets = get_batch(
                data,
                batch_size=2,
                block_size=3,
                device=torch.device("cpu"),
            )

        self.assertEqual(inputs.tolist(), [[10, 11, 12], [20, 21, 0]])
        self.assertEqual(targets.tolist(), [[11, 12, 13], [21, 22, -100]])

    def test_raises_instead_of_truncating_example_exceeding_block_size(self) -> None:
        data = [torch.tensor([1, 2, 3, 4])]
        with patch("train.torch.randint", return_value=torch.tensor([0])):
            with self.assertRaisesRegex(ValueError, "exceeding block-size=2"):
                get_batch(
                    data,
                    batch_size=1,
                    block_size=2,
                    device=torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()
