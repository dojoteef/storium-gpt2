"""
This file runs a number of commands including data preprocessing
"""
import argparse
import logging
from typing import Any, Dict
from types import SimpleNamespace

# import train first, so that comet initializes before torch
from train import define_train_args
from evaluate import define_eval_args
from generate import define_generate_args
from data.dataset import define_preprocess_args
from data.preprocess import define_split_args


def parse_args() -> SimpleNamespace:
    """ Parse the arguments required for splitting the dataset """
    parser = argparse.ArgumentParser("Storyteller")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Where to cache the downloaded pretrained config/tokenizer/model",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="""
Location of the data. For preprocessing provide raw data directory. For
training provide directory of preprocessed data.
""",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Where to output files. File types generate depend on command.",
    )
    sub_parsers = parser.add_subparsers()
    define_split_args(sub_parsers)
    define_preprocess_args(sub_parsers)
    define_eval_args(sub_parsers)
    define_train_args(sub_parsers)
    define_generate_args(sub_parsers)

    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        action="count",
        help="Whether to have verbose output",
    )

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_usage()
        exit(1)

    # pylint:disable=protected-access
    def extract(parser) -> Dict[str, Any]:
        """
        Extract the named from the parser
        """
        namespaces = {}
        for group in parser._action_groups:
            if not group.title:
                raise ValueError("Must specify a title for the subgroup!")

            group_dict = {
                a.dest: getattr(args, a.dest, None) for a in group._group_actions
            }
            if all(v is None for v in group_dict.values()):
                continue

            if (
                group.title == "optional arguments"
                or group.title == "positional arguments"
            ):
                namespaces.update(group_dict)
            else:
                namespaces[group.title] = SimpleNamespace(**group_dict)

        return namespaces

    # pylint:enable=protected-access

    namespaces = {}
    for sub_parser in sub_parsers.choices.values():
        if args.func != sub_parser.get_default("func"):
            continue

        extracted = extract(sub_parser)
        if not extracted:
            raise ValueError("Unable to extract sub_parser arguments!")

        namespaces.update(extracted)
        namespaces.update(
            {
                k: getattr(args, k, None)
                for k, v in sub_parser._defaults.items()  # pylint:disable=protected-access
            }
        )
    namespaces.update(extract(parser))

    return SimpleNamespace(**namespaces)


def main():
    """
    Main entry point when calling this module directly
    """
    args = parse_args()

    # pylint appears confused, so disable these warnings
    # pylint:disable=no-member
    if args.verbose:
        logging.getLogger().setLevel(
            logging.DEBUG if args.verbose > 1 else logging.INFO
        )

    # perform the subcommand chosen
    args.func(args)
    # pylint:enable=no-member


if __name__ == "__main__":
    main()
