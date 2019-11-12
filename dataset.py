"""
Utilities and classes for manipulating the dataset
"""
import os
import logging
import argparse
from typing import Dict, List

from transformers import (
    PreTrainedTokenizer,
    BertTokenizer,
    OpenAIGPTTokenizer,
    GPT2Tokenizer,
    TransfoXLTokenizer,
    XLNetTokenizer,
    XLMTokenizer,
    RobertaTokenizer,
    DistilBertTokenizer,
)
import torch
from torch.utils.data import Dataset

from preprocess import process_story_file, get_entry, split_dataset


TOKENIZER_NAMES: Dict[type, str] = {
    BertTokenizer: "bert",
    OpenAIGPTTokenizer: "gpt",
    GPT2Tokenizer: "gpt2",
    TransfoXLTokenizer: "transfo-xl",
    XLNetTokenizer: "xlnet",
    XLMTokenizer: "xlm",
    RobertaTokenizer: "roberta",
    DistilBertTokenizer: "distilbert",
}


class StoriumDataset(Dataset):
    """
    The torch dataset class for Storium for use in a DataLoader
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.entries: List[torch.Tensor] = []

    def process_and_load(
        self, filenames: List[str], directory: str, prefix: str, force: bool = False
    ):
        """
        Process the stories from the file list and generate the tensors for the
        dataset

        - **filenames**: A list of filenames to preprocess
        - **directory**: path to a directory to save the result
        - **prefix**: a prefix for the filename to generate
        - **force**: whether to force processing if the target file already exists

        Returns the path of the preprocessed file to load
        """
        tokenizer_name = TOKENIZER_NAMES.get(type(self.tokenizer), "unknown")
        if tokenizer_name == "unknown":
            logging.warning("Unknown tokenzier used!")

        output_path = os.path.join(directory, f"{prefix}.{tokenizer_name}.pt")
        if os.path.isfile(output_path) and not force:
            self.entries = torch.load(output_path)
            return

        self.entries = []
        for filename in filenames:
            story = process_story_file(filename, tokenizer=self.tokenizer)
            if not story:
                logging.warning("Skipped %s", filename)
                continue

            for entry_id in story.entries.keys() if story else []:
                self.entries.append(
                    torch.tensor(  # pylint:disable=not-callable
                        get_entry(story, entry_id)
                    )
                )

        torch.save(self.entries, output_path)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]


def parse_args():
    """ Parse the arguments required for splitting the dataset """
    parser = argparse.ArgumentParser("Split Dataset")
    parser.add_argument(
        "data_directory",
        type=str,
        help="Location of the raw dataset (directory of json files)",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        default=".",
        help="Where to output the split files",
    )
    parser.add_argument(
        "--train-split",
        type=int,
        default=8,
        help="An int denoting the relative amount of data to use for training",
    )
    parser.add_argument(
        "--validation-split",
        type=int,
        default=1,
        help="An int denoting the relative amount of data to use for validation",
    )
    parser.add_argument(
        "--test-split",
        type=int,
        default=1,
        help="An int denoting the relative amount of data to use for testing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        action="count",
        help="Whether to have verbose output",
    )

    return parser.parse_args()


def main():
    """
    Main entry point when calling this module directly
    """
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(
            logging.DEBUG if args.verbose > 1 else logging.INFO
        )

    splits = split_dataset(
        args.data_directory, (args.train_split, args.validation_split, args.test_split)
    )
    for split, filenames in zip(("train", "validation", "test"), splits):
        with open(
            os.path.join(args.output_directory, f"{split}_filenames.txt"), "wt"
        ) as split_file:
            split_file.write("\n".join(filenames))


if __name__ == "__main__":
    main()
