"""
Generate from our Storium models
"""
import os
import sys
import json
import logging
import argparse
from typing import Any, Dict, List, Tuple

import torch
from torch.nn import functional as F
from transformers import GPT2Config, PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from data.dataset import StoriumDataset
from data.parallel import StaticDataParallel
from data.preprocess import SPLIT_NAMES, SpecialToken
from data.utils import collate, EntryList, narrow
from model import GPT2SegmentedModel
from utils import grouper, tqdm_wrap_stdout


class Generator:
    """
    A class that encapsulates all the functionality needed to generate from a model
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the generator
        """
        self.args = args
        self.move_id: int
        self.separator_id: int
        self.model: PreTrainedModel
        self.dataset: StoriumDataset
        self.tokenizer: PreTrainedTokenizer
        self.train_args: argparse.Namespace

    def load(self, checkpoint_dir):
        """
        Load the model and tokenizer from the specified path
        """
        logging.info("Loading train config")
        train_config_filename = os.path.join(checkpoint_dir, "train_config.json")
        if not os.path.isfile(train_config_filename):
            raise RuntimeError(
                f"Cannot find train config file: {train_config_filename}"
            )

        # Must load the train config first
        with open(train_config_filename, "rt") as config_file:
            self.train_args = json.load(
                config_file, object_hook=lambda obj: argparse.Namespace(**obj)
            )

        logging.info("Loading model")
        config = GPT2Config.from_pretrained(checkpoint_dir)
        config.output_past = True

        model = GPT2SegmentedModel.from_pretrained(
            checkpoint_dir, config=config, cache_dir=self.args.cache_dir
        )

        if torch.cuda.is_available():
            model = model.cuda()

        self.model = StaticDataParallel(model)

    def load_dataset(self, split: str):
        """
        Load the dataset
        """
        if not hasattr(self, "dataset") or self.dataset.split != split:
            logging.info("Loading %s dataset", split)
            self.dataset = StoriumDataset(
                split, self.train_args.model.model_name, cache_dir=self.args.cache_dir,
            )
            self.dataset.load(self.args.data_dir)
            self.tokenizer = self.dataset.get_tokenizer()
            self.move_id = self.tokenizer.convert_tokens_to_ids(SpecialToken.move)
            self.separator_id = self.tokenizer.convert_tokens_to_ids(
                SpecialToken.separator
            )

    def filter(self, logits):
        """
        Do top-k/top-p filtering of the logits

        Based on: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        top_k = min(self.args.sample.top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float("inf")

        top_p = self.args.sample.top_p
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float("inf")

        return logits

    def extract_summary(self, entries: EntryList) -> List[Dict[str, Any]]:
        """
        Extract the summaries from a list of entries by truncating the final
        segment in the entry, which is the move itself
        """
        entry_list: List[Dict[str, Any]] = []
        for entry in entries:
            # Use the index of the last separator to truncate the entry
            indices = (entry["tokens"] == self.separator_id).nonzero().flatten()
            entry_list.append(narrow(entry, indices[-1] + 1))

        return entry_list

    def sample_move(
        self, entries: EntryList, length: int = 256, max_length: int = 1024
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Sample entry text given the list of summaries up to the specified length
        """
        summary = self.extract_summary(entries)

        with torch.no_grad():
            self.model.eval()
            batch = collate(summary)
            lengths = batch["lengths"]
            seq_length = batch["tokens"].shape[1]
            batch = self.sample_first(batch)

            done = [t.item() == self.separator_id for t in batch["tokens"]]
            outputs = [t.tolist() for t in batch["tokens"]]
            for _ in range(seq_length, min(seq_length + length, max_length)):
                batch = self.sample_next(batch)
                for idx, (token, output) in enumerate(zip(batch["tokens"], outputs)):
                    next_token = token.item()
                    done[idx] = done[idx] or (next_token == self.separator_id)
                    if done[idx]:
                        continue

                    output.append(next_token)

        return (
            [self.tokenizer.decode(e["tokens"].tolist()) for e in summary],
            [
                self.tokenizer.decode(e["tokens"][l:].tolist())
                for e, l in zip(entries, lengths)
            ],
            [self.tokenizer.decode(output) for output in outputs],
        )

    def sample_logits(self, logits) -> torch.Tensor:
        """
        Perform sampling on the passed in logits
        """
        temperature = self.args.sample.temperature
        logits = self.filter(logits / (temperature if temperature > 0 else 1.0))
        return (
            torch.argmax(logits, dim=-1)
            if temperature == 0  # greedy sampling
            else torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)
        )

    def sample_first(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sample the first token. The first token is a special case since we pass
        in the full context and generate the "past" hidden states.
        """
        lengths = batch["lengths"]
        num_tokens = batch["num_tokens"]

        batch_size = len(lengths)
        indices = torch.LongTensor(lengths)  # type:ignore

        outputs = self.model(batch)
        next_token = self.sample_logits(outputs[0][range(batch_size), indices - 1])

        # Return an updated batch
        return {
            "past": outputs[1],
            "tokens": next_token.unsqueeze(-1),
            "segments": next_token.new_full((batch_size, 1, 1), self.move_id),
            "segment_masks": next_token.new_ones((batch_size, 1, 1)),
            "lengths": [l + 1 for l in lengths],
            "num_tokens": num_tokens + batch_size,
        }

    def sample_next(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sample the next token for the passed in batch
        """
        lengths = batch["lengths"]
        num_tokens = batch["num_tokens"]

        batch_size = len(lengths)

        outputs = self.model(batch)
        next_token = self.sample_logits(outputs[0][:, 0])

        # Return an updated batch
        return {
            "past": outputs[1],
            "tokens": next_token.unsqueeze(-1),
            "segments": batch["segments"],
            "segment_masks": batch["segment_masks"],
            "lengths": [l + 1 for l in lengths],
            "num_tokens": num_tokens + batch_size,
        }

    def __call__(self):
        """
        Run the generation!
        """
        entries = self.dataset.entries
        if self.args.data.max_entries:
            entries = entries[: self.args.data.max_entries]

        batch_iterator = tqdm(
            grouper(entries, self.args.data.batch_size),
            unit="batch",
            initial=1,
            dynamic_ncols=True,
            desc="Generating",
            file=sys.stdout,  # needed to make tqdm_wrap_stdout work
        )

        sep = "*******\n"
        with tqdm_wrap_stdout():
            for batch_idx, batch in enumerate(batch_iterator):
                contexts, originals, samples = self.sample_move(batch)
                for idx, (context, original, sample) in enumerate(
                    zip(contexts, originals, samples)
                ):
                    logging.info(
                        "#%d:\n%scontext\n%s%s\n%soriginal\n%s%s\n%ssample\n%s%s",
                        batch_idx + idx,
                        sep,
                        sep,
                        context,
                        sep,
                        sep,
                        original,
                        sep,
                        sep,
                        sample,
                    )

            batch_iterator.close()


def define_generate_args(
    sub_parsers: argparse._SubParsersAction,  # pylint:disable=protected-access
):
    """
    Define arguments needed for the evaluation command
    """
    parser = sub_parsers.add_parser("generate", help="Generate samples from a model")
    parser.add_argument(
        "--restore",
        type=str,
        help="Restore from the specified checkpoint before evaluation",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=SPLIT_NAMES,
        help="Which dataset split to run the evaluation over",
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of examples to batch together",
    )
    data_group.add_argument(
        "--max-entries",
        type=int,
        default=0,
        help="If greater than 0, then only process up to max entries",
    )

    sample_group = parser.add_argument_group("sample")
    sample_group.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="top_k > 0: keep only top k tokens with highest probability",
    )
    sample_group.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="top_p > 0.0: keep the top tokens with cumulative probability >= top_p"
        "Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)",
    )
    sample_group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature == 0.0: greedy decoding; temperature == 1.0: normal multinomial samples",
    )

    parser.set_defaults(func=perform_generation)


def perform_generation(args):
    """
    Main entry point for generation
    """
    generator = Generator(args)
    generator.load(args.restore)
    generator.load_dataset(args.split)
    generator()
