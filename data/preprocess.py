"""
This file contains a number of utility methods for preprocessing stories.
"""
import os
import glob
import json
import heapq
import bisect
import logging
import argparse
from numbers import Number
from dataclasses import dataclass
from itertools import islice, zip_longest
from contextlib import contextmanager

from enum import Enum, auto, unique
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch
from kiwisolver import Constraint, Solver, Variable, strength
from transformers import PreTrainedTokenizer

SPLIT_NAMES = ("train", "validation", "test")


###############################################################################
# IDEA:
# Make all the summarize_* methods take in a dropout probability, such that
# they randomly dropout a component as SpecialToken.missing, but the loss is
# still calculated against the full sequence, such that it can learn to fill in
# missing data.
#
# This, for example, could be used as a simple technique for predicting a card
# to play for the "wild" cards. Could use the BART model for this.
###############################################################################


@unique
class SpecialToken(str, Enum):
    """
    An enumeration of special tokens
    """

    # This method must be defined before calling "auto()"
    # pylint:disable=unused-argument,no-self-argument
    def _generate_next_value_(name, start, count, last_values):
        """
        Automatically create the enumeration name
        """
        return f"<|{name.upper()}|>"

    # pylint:enable=unused-argument,no-self-argument

    @classmethod
    def from_string(cls, name: str):
        """
        Get the associated SpecialToken from the passed in string
        """
        return cls(f"<|{name.upper()}|>")

    missing = auto()  # denotes missing information
    character = auto()  # a character's biography

    # All the possible card types from <CardNamespace> in the Storium export
    # format: https://storium.com/help/export/json/0.9.2
    chartype = auto()
    goal = auto()
    person = auto()
    place = auto()
    thing = auto()
    strength = auto()
    weakness = auto()
    obstacle = auto()
    subplot = auto()

    # Contextual card attributes
    failure_stakes = auto()
    success_stakes = auto()

    # Information denoting entry type
    move = auto()
    establishment = auto()
    addition = auto()
    conclusion = auto()

    # some notions of ordering
    current = auto()
    previous = auto()

    # some notions of authorship
    narrator = auto()  # can stack,  e.g. previous + narrator
    same_character = auto()  # can stack,  e.g. previous + same_character
    diff_character = auto()  #  can stack,  e.g. previous + diff_character

    def __str__(self):
        """
        Override the default string method to return the enumeration value,
        which is a string
        """
        return self.value


class Trim(Enum):
    """
    An enum denoting how to trim a Segment that is too long

    - **start**: trim the start of the segment
    - **end**: trim the end of the segment
    - **middle**: trim the middle of the segment
    """

    start = auto()
    end = auto()
    middle = auto()


class Segment(tuple):
    """
    This structure holds a sequence of token ids from the tokenizer vocabulary, along with a
    list of segment ids that modify the entire sequence of token ids.
    """

    # metadata
    segment_ids: Tuple[int, ...]

    # Layout related variables
    trim: Trim
    constrained: bool
    naive_max_length: int
    preferred_length: int
    length: Optional[Variable]

    def __new__(
        cls,
        *iterable: Union[Iterable[int], Iterable["Segment"]],
        segment_ids: Iterable[int] = tuple(),
        preferred_length: int = 0,
        trim: Trim = Trim.end,
    ):
        """
        Create a Segment
        """
        self = super().__new__(cls, (*iterable))  # type: ignore

        # Initialize the layout related variables before iterating over the
        # newly created tuple. It must be unconstrained to begin with.
        self.trim = trim
        self.constrained = False
        self.naive_max_length = -1
        self.preferred_length = preferred_length

        all_ints = all(isinstance(t, int) for t in self)
        if not (all_ints or all(isinstance(t, Segment) for t in self)):
            raise ValueError(
                f"{cls.__name__}() only accepts a homogenous sequence of int or {cls.__name__}"
            )

        # If the segment only contains tokens ids, then it can be constrained.
        # Have to additionally include the check that the unconstrained
        # length is positive, because python's all() returns True for empty
        # collections.
        self.length = Variable() if all_ints and self.unconstrained_length else None

        self.segment_ids = tuple(segment_ids)
        if not all(isinstance(t, int) for t in segment_ids):
            raise ValueError(f"{cls.__name__}() segment_ids must be int!")

        return self

    @property
    def constraints(self) -> List[Constraint]:
        """
        Return a list of constraints defining the length of the segment
        """
        constraints: List[Constraint] = []
        for segment in self:
            if isinstance(segment, Segment):
                constraints.extend(segment.constraints)

        if self.constrained and self.length:
            # Get the underlying length of the data
            length = self.unconstrained_length

            # Cannot be shorter than 1
            constraints.append((self.length >= 1) | strength.required)

            # Cannot be longer than the underlying length
            constraints.append((self.length <= length) | strength.required)

            # Try to stay close to the underlying length
            constraints.append((self.length == length) | strength.medium)

            # Resist shrinking below underlying length
            constraints.append((self.length >= length) | strength.medium)

            # Strongly try to stay close to the preferred length
            if self.preferred_length:
                constraints.append((self.length == length) | strength.strong)

        return constraints

    def __len__(self):
        """
        By default length returns the constrained length. To get the full
        length see Segement.unconstrained_length
        """
        unconstrained_length = self.unconstrained_length
        if self.naive_max_length >= 0:
            if not self.preferred_length:
                return min(unconstrained_length, 100)

            return unconstrained_length

        if not self.constrained or not self.length:
            return unconstrained_length

        return int(self.length.value()) or unconstrained_length

    def __getitem__(self, key):
        """
        Allow __getitem__ to support constraining the underlying sequence
        """
        return super().__getitem__(self.constrained_slice)[key]

    def __iter__(self):
        """
        Iterate over the constrained Segment
        """
        return iter(super().__getitem__(self.constrained_slice))

    @property
    def constrained_slice(self) -> slice:
        """
        Compute the constrained slice for the Segment
        """
        length = len(self)
        if self.trim is Trim.end:
            return slice(length)

        if self.trim is Trim.start:
            return slice(-length, None)

        if self.trim is Trim.middle:
            remaining = self.unconstrained_length - length
            start = remaining // 2
            end = start - remaining
            return slice(start, end)

        raise RuntimeError("Unknown trim type!")

    @property
    def unconstrained_length(self):
        """
        Get the full unconstrained length of the Segment
        """
        return super().__len__()

    @property
    def token_segments(self):
        """
        A generator which yields all token ids within the segment recursively,
        along with their associated segment ids, which respects max length.
        """
        if self.constrained and self.naive_max_length >= 0:
            return islice(self._token_segments, self.naive_max_length)

        return self._token_segments

    @property
    def _token_segments(self):
        """
        A generator which yields all token ids within the segment recursively,
        along with their associated segment ids
        """
        for segment in self:
            if isinstance(segment, Segment):
                for (
                    token_id,
                    segment_ids,
                ) in segment._token_segments:  # pylint:disable=protected-access
                    yield token_id, self.segment_ids + segment_ids
            else:
                yield segment, self.segment_ids

    @property
    def lengths(self):
        """
        A generator which yields all the length variables within the segment recursively
        """
        for segment in self:
            if isinstance(segment, Segment):
                yield from segment.lengths

        if self.length:
            yield self.length

    def _mark_constrained(self, constrained: bool, max_length: int = -1):
        """
        Internal method to mark the Segment hierarchy as constrained or not.

        DO NOT CALL THIS DIRECTLY!
        """
        for segment in self:
            if isinstance(segment, Segment):
                segment._mark_constrained(  # pylint:disable=protected-access
                    constrained, max_length
                )

        self.constrained = constrained
        self.naive_max_length = max_length

    def _constrain(self, max_length: int):
        """
        Internal method that constrains the Segment to a maximum length using kiwisolver.

        DO NOT CALL THIS DIRECTLY!
        """
        solver = Solver()
        for constraint in self.constraints:
            solver.addConstraint(constraint)

        solver.addConstraint(sum(self.lengths) <= max_length)
        solver.updateVariables()

    @contextmanager
    def constraint(self, max_length: int, naive: bool = False):
        """
        A context manager that constrains the Segment, performs the necessary
        operation then unconstrains the Segment.
        """
        naive_max_length = max_length if naive else -1
        self._mark_constrained(True, naive_max_length)
        self._constrain(max_length)
        yield
        self._mark_constrained(False)

    def asdict(self, *, with_stats: bool = False) -> Dict[str, Any]:
        """
        Convert a sequence of annotated tokens into a dictionary of the form
        (where "stats" is optional):

        {
            "tokens": Tuple[int, ...],
            "segments": Tuple[
                {
                    "mask": Tuple[float, ...],
                    "values": Tuple[int, ...],
                },
                ...
            ]
            "stats": {
                int: count,
                ...
            }
        }
        """
        tokens, segments = zip(*self.token_segments)
        segment_dict = {
            "tokens": tokens,
            "segments": tuple(
                {
                    "mask": tuple(0.0 if s < 0 else 1.0 for s in segment),
                    # Cannot have a negative index, so just set it to 0, since
                    # it's going to get masked out anyway
                    "segments": tuple(0 if s < 0 else s for s in segment),
                }
                for segment in zip_longest(*segments, fillvalue=-1)
            ),
        }

        if with_stats:
            stats: Dict[int, int] = {}
            for _, segment_ids in self.token_segments:
                for segment_id in set(segment_ids):
                    stats[segment_id] = stats.get(segment_id, 0) + 1
            segment_dict["stats"] = stats

        return segment_dict


DataType = TypeVar("DataType")


class IndexedSet(List[DataType]):
    """
    A class that makes indexing a unique sorted list easy. All the entries must
    have unique keys, if you try to insert an already existing key, it will
    raise an error.

    Loosely based on SortedCollection, which is referenced in the python docs
    for bisect.

    See: https://code.activestate.com/recipes/577197-sortedcollection/
    """

    def __init__(self, *iterable: Iterable[DataType], key=int):
        # Ensure the list is in sorted order
        self._key = key

        values = tuple(*iterable)
        if values:
            keys, values = zip(*sorted((key(i), i) for i in values))
            self._keys = list(keys)
        else:
            self._keys = []

        super().__init__(values)

    def insert(self, value):
        """
        Insert into the set
        """
        key = self._key(value)
        idx = bisect.bisect_left(self._keys, key)
        if (
            idx != len(self._keys)
            and self[idx] == value  # pylint:disable=unsubscriptable-object
        ):
            # it's already in the set, no need to insert it
            return

        self._keys.insert(idx, key)
        super().insert(idx, value)  # pylint:disable=no-member

    def index(self, value: DataType) -> int:  # type: ignore
        """
        Find the index of the item in the set
        """
        key = self._key(value)
        idx = bisect.bisect_left(self._keys, key)
        if (
            idx != len(self._keys)
            and self[idx] == value  # pylint:disable=unsubscriptable-object
        ):
            return idx

        raise ValueError(f"{value} not in set")


@dataclass
class CharacterInfo:
    """
    The processed character info
    """

    summary: Segment
    character_id: str

    # This is a sorted list of entry ids written by the character to
    # allow easily looking up the previous entries for the character
    entry_ids: IndexedSet


@dataclass
class EntryInfo:
    """
    The processed entry info
    """

    entry_id: str
    character_id: str
    establishment_id: str
    text: Segment
    summary: Segment


class IndexedDict(Dict[str, DataType]):
    """
    A convenient wrapper around dict that allows integer-based indexing
    operations
    """

    indices: List[str]
    reverse_indices: Dict[str, int]

    def __init__(
        self, mapping: Union[Iterable[Tuple[str, DataType]], Mapping[str, DataType]],
    ):
        super().__init__()
        self.indices = []
        self.reverse_indices = {}

        if isinstance(mapping, Mapping):
            mapping = mapping.items()

        for idx, (key, value) in enumerate(mapping):
            self.indices.append(key)
            self.reverse_indices[key] = idx
            super().__setitem__(key, value)  # pylint:disable=no-member

    def __delitem__(self, key: str):
        raise RuntimeError("IndexedDict is immutable!")

    def __setitem__(self, key: str, value: DataType):
        raise RuntimeError("IndexedDict is immutable!")

    def index(self, key: str) -> int:
        """
        Return the index of the key
        """
        return self.reverse_indices[key]


@dataclass
class ProcessedStory:
    """
    This defines the structure of a story after processing
    """

    game_id: str

    # A mapping of character id to character info
    characters: IndexedDict[CharacterInfo]

    # A mapping of entry id to entry info
    entries: IndexedDict[EntryInfo]

    # A mapping of entry id to establishment's entry info
    establishment_entries: IndexedDict[EntryInfo]


def extract_string(field: str, mapping: Dict[str, Any]) -> str:
    """
    Extract the given string field, accounting for the potential that it is
    specified as None
    """
    return mapping.get(field, SpecialToken.missing) or SpecialToken.missing


class Preprocessor:
    """
    This class encapsulates the functionality needed to preprocess the Storium
    data in a format that we can use.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        history: int = 0,
        character_history: int = 0,
        max_length: int = 1024,
        preferred_entry_length: int = 256,
    ):
        """
        NOTE: The default max_length parameter was chosen to support GPT-2.
        """
        self.tokenizer = tokenizer

        self.history = history
        self.character_history = character_history

        self.max_length = max_length
        self.preferred_entry_length = preferred_entry_length

    def encode(self, string_or_list: Union[str, List[str]]) -> List[int]:
        """
        PreTrainedTokenizer.encode outputs warnings if the text being tokenized
        is longer than the max_length specified in the tokenizer.
        Unfortunately, the order of operations is to warn first, then truncate
        to the max length that was passed in, resulting in spurious warnings,
        so we wrap the function to suppress these warning messages.
        """
        logger = logging.getLogger(PreTrainedTokenizer.__module__)
        log_level = logger.getEffectiveLevel()
        logger.setLevel(logging.ERROR)

        tokens = self.tokenizer.encode(
            " ".join(string_or_list)
            if isinstance(string_or_list, list)
            else string_or_list,
            max_length=self.max_length,
        )
        logger.setLevel(log_level)
        return tokens

    def encode_special(
        self,
        string_or_list: Union[str, List[str]],
        special_tokens: Iterable[SpecialToken] = tuple(),
        preferred_length: int = 0,
        trim: Trim = Trim.end,
    ) -> Segment:
        """
        After encoding with the tokenizer, this creates, create a Segment and
        assign the special_tokens if specified.
        """
        return Segment(
            self.encode(string_or_list),
            segment_ids=tuple(self.tokenizer.convert_tokens_to_ids(special_tokens))
            if special_tokens
            else tuple(),
            preferred_length=preferred_length,
            trim=trim,
        )

    def summarize_character(self, character: Dict[str, Any]) -> Segment:
        """
        Create the summary for a character
        """
        strings: List[str] = []
        if character:
            strings.append(extract_string("name", character))
            strings.append(extract_string("description", character))
        else:
            strings.append(SpecialToken.missing)

        return self.encode_special(strings, (SpecialToken.character,))

    def summarize_cards(self, cards: List[Dict[str, Any]],) -> Segment:
        """
        Create the summary of a card
        """
        return Segment(
            # we are not passing null for the card, so we can ignore mypy's
            # concern that we may be adding a null value
            (
                self.summarize_card(  # type: ignore
                    card
                )
                for card in cards
            )
        )

    def summarize_card(self, card: Optional[Dict[str, Any]],) -> Optional[Segment]:
        """
        Create the summary of a card.

        If it's a challenge card, then it'll have "success_stakes" and
        "failure_stakes" as well.
        """
        if not card:
            return None

        summary = [
            self.encode_special(
                [extract_string("name", card), extract_string("description", card)],
            )
        ]

        for field in ("success_stakes", "failure_stakes"):
            if card.get(field):
                summary.append(
                    self.encode_special(
                        extract_string(field, card), (SpecialToken.from_string(field),),
                    )
                )

        return Segment(
            summary,
            segment_ids=tuple(
                self.tokenizer.convert_tokens_to_ids(
                    (SpecialToken.from_string(card["namespace"]),)
                ),
            ),
        )

    def summarize_entry(self, entry: Dict[str, Any]) -> Optional[Segment]:
        """
        Create the summary of an entry
        """
        if not entry:
            return None

        summary = []
        entry_type = entry["format"]
        if entry_type == "move":
            challenge = self.summarize_card(entry.get("target_challenge_card"),)
            if challenge:
                summary.append(challenge)

            cards = self.summarize_cards(entry.get("cards_played_on_challenge", []),)
            if cards:
                summary.append(cards)
        elif entry_type == "establishment":
            place = self.summarize_card(entry.get("place_card"))
            if place:
                summary.append(place)
        elif entry_type == "addition":
            cards = self.summarize_cards(entry.get("challenge_cards", []),)
            if cards:
                summary.append(cards)

        return Segment(
            summary,
            segment_ids=tuple(
                self.tokenizer.convert_tokens_to_ids(
                    (SpecialToken.from_string(entry_type),)
                ),
            ),
        )

    def process_entry(
        self, entry: Dict[str, Any], establishment_id: str,
    ) -> Optional[EntryInfo]:
        """
        Process a character entry
        """
        text = entry.get("description", "")
        if not text and entry.get("format") != "establishment":
            # Only modeling moves with written text, though make a special
            # exception for establishment entries. While they are currently
            # required to have text, it seems at some point there were games that
            # didn't have any text for the establishment entry, though it would still
            # have place cards.
            return None

        encoded_text = self.encode_special(
            text,
            (SpecialToken.from_string(entry["format"]),),
            preferred_length=self.preferred_entry_length,
        )
        summary = self.summarize_entry(entry)
        if not summary:
            summary = self.encode_special(
                text,
                (SpecialToken.from_string(entry["format"]),),
                trim=Trim.start,  # Treat the end of the entry text as a summary
            )

        return EntryInfo(
            entry_id=entry["seq_id"],
            character_id=entry["role"],
            establishment_id=establishment_id,
            text=encoded_text,
            summary=summary,
        )

    def process_story_file(self, filename: str) -> Optional[ProcessedStory]:
        """
        Wrapper around process_story that takes in a filename of a story to process
        """
        with open(filename, "rt") as file:
            story = json.load(file)

        return self.process_story(story)

    def process_story(self, story: Dict[str, Any]) -> Optional[ProcessedStory]:
        """
        Summarize a scene. Returns a list of summarized scene entries.
        """
        scenes = story.get("scenes")
        characters = story.get("characters")
        if not scenes or not characters or not isinstance(scenes, Sequence):
            return None

        character_list = [
            # Treat narrator as a character who is always present without a summary
            (
                "narrator",
                CharacterInfo(
                    entry_ids=IndexedSet(), character_id="narrator", summary=Segment(),
                ),
            )
        ]
        for character in characters:
            character_id = character.get("character_seq_id")
            if not character_id:
                # This would only happen with malformed data
                continue

            # The <Character> object as defined in the Storium export format only
            # has the `character_seq_id` field, but what we really want to use is
            # the `role` field from the <Entry> object. But it's easy enough to
            # create the specified role as it's simply defined as:
            #
            # <RoleString> = string
            #   either 'narrator' or 'character:XYZ' where XYZ is the <CharacterSeqId>
            # See https://storium.com/help/export/json/0.9.2
            character_id = f"character:{character_id}"
            character_list.append(
                (
                    character_id,
                    CharacterInfo(
                        entry_ids=IndexedSet(),
                        character_id=character_id,
                        summary=self.summarize_character(character),
                    ),
                )
            )

        all_characters = IndexedDict(character_list)
        entry_list: List[Tuple[str, EntryInfo]] = []
        establishment_list: List[Tuple[str, EntryInfo]] = []
        for scene in scenes:
            entries = scene.get("entries", [])
            if not entries or not isinstance(entries, Sequence):
                continue

            for entry in entries:
                entry_id = entry.get("seq_id", None)
                if entry_id is None:
                    # This would only happen with malformed data
                    continue

                entry_format = entry.get("format")
                entry_info = self.process_entry(
                    entry, establishment_list[-1][0] if establishment_list else entry_id
                )
                if not entry_info:
                    continue

                entry_list.append((entry_id, entry_info))
                if entry_format == "establishment":
                    establishment_list.append((entry_id, entry_info))

                # See https://github.com/PyCQA/pylint/issues/3129
                character_info = all_characters[  # pylint:disable=unsubscriptable-object
                    entry["role"]
                ]
                character_info.entry_ids.insert(entry_id)

        return ProcessedStory(
            game_id=story["game_pid"],
            entries=IndexedDict(entry_list),
            characters=all_characters,
            establishment_entries=IndexedDict(establishment_list),
        )

    def get_entry_summary(self, story: ProcessedStory, entry_id: str) -> Segment:
        """
        Extracts the context for a given entry
        """
        game_id = story.game_id
        entry_info = story.entries.get(entry_id)
        if not entry_info:
            raise KeyError(f"Cannot find entry {entry_id} for story {game_id}!")

        character_id = entry_info.character_id
        character_info = story.characters.get(character_id)
        if not character_info:
            raise KeyError(f"Cannot find character {character_id} for story {game_id}!")

        summary = [character_info.summary]
        establishment_id = entry_info.establishment_id
        if entry_id != establishment_id:
            establishment_info = story.establishment_entries.get(
                entry_info.establishment_id
            )
            if not establishment_info:
                raise KeyError(
                    f"""
                    Cannot find establishment entry {establishment_id} for story {game_id}!
                    """
                )
            summary.append(establishment_info.summary)

        end_index = max(0, character_info.entry_ids.index(entry_id) - 1)
        start_index = max(0, end_index - self.character_history)
        prev_entry_ids = IndexedSet(
            character_info.entry_ids[idx] for idx in range(start_index, end_index)
        )

        end_index = max(0, story.entries.index(entry_id) - 1)
        start_index = max(0, end_index - self.history)
        for entry_index in range(start_index, end_index):
            prev_entry_ids.insert(story.entries.indices[entry_index])

        for idx, prev_entry_id in enumerate(prev_entry_ids, 1):
            # Get the summary of the previous entries from the same character and
            # mark them as such
            entry_info = story.entries[prev_entry_id]
            if entry_info.entry_id == entry_info.establishment_id:
                continue

            character_token = (
                SpecialToken.same_character
                if entry_info.character_id == character_id
                else SpecialToken.diff_character
            )
            summary.append(
                Segment(
                    entry_info.summary,
                    segment_ids=tuple(
                        self.tokenizer.convert_tokens_to_ids(
                            idx * (SpecialToken.previous,) + (character_token,)
                        )
                    ),
                )
            )

        entry_summary = entry_info.summary
        if prev_entry_ids:
            # If we used any history, then mark this entry as the current entry
            entry_summary = Segment(
                entry_info.summary,
                segment_ids=tuple(
                    self.tokenizer.convert_tokens_to_ids(
                        (SpecialToken.current, SpecialToken.same_character)
                    )
                ),
            )
        summary.append(entry_summary)

        return Segment(summary)

    def get_move(self, story: ProcessedStory, entry_id: str) -> Segment:
        """
        Extracts the context for a given entry
        """
        entry_info = story.entries.get(entry_id)
        if not entry_info:
            raise KeyError(f"Cannot find entry {entry_id} for story {story.game_id}!")

        if entry_info.character_id == "narrator":
            # Only characters can do a normal "move"
            return Segment()

        return Segment((self.get_entry_summary(story, entry_id), entry_info.text))


def tensorize(
    nested: Union[Sequence, Mapping]
) -> Union[Sequence, Mapping, torch.Tensor]:
    """
    Convert the potentially nested sequence or mapping of ints to torch tensors
    """
    if not nested:
        return nested

    if isinstance(nested, Sequence):
        element = nested[0]
        if isinstance(element, Number):
            return torch.tensor(nested)  # pylint:disable=not-callable
        elif isinstance(element, Sequence) or isinstance(element, Mapping):
            return type(nested)(tensorize(e) for e in nested)  # type: ignore
    elif isinstance(nested, Mapping):
        element = next(iter(nested.values()))
        if isinstance(element, Sequence) or isinstance(element, Mapping):
            return {k: tensorize(v) for k, v in nested.items()}

    return nested


def split_dataset(data_path, splits: Tuple[int, ...]) -> Tuple[List[str], ...]:
    """
    Return a list of files that split the dataset according to these constraints:

    1) They approximately meet the passed in splits, which are treated as
    ratios by taking splits[i]/sum(splits)
    2) We balance the number of stories and token counts for each split
    according to the ratios

    NOTE: The number of words in an entry is based upon splitting along
    whitespace boundaries. This is to divorce the dataset splits from any
    particular tokenization scheme, e.g. GPT2
    """
    story_info = []
    total_files = 0.0
    total_words = 0.0
    total_scenes = 0.0
    total_entries = 0.0
    data_path = os.path.abspath(data_path)
    for filename in glob.glob(os.path.join(data_path, "**/*.json"), recursive=True):
        with open(filename, "rt") as file:
            story = json.load(file)

        num_words = 0
        num_entries = 0
        scenes = story.get("scenes", [])
        for scene in scenes:
            entries = scene.get("entries", [])
            num_entries += len(entries)
            for entry in entries:
                description = entry.get("description", "") or ""

                # We do a very simple tokenization that simply splits on whitespace to
                # get a ballpark estimate of the length of the story, only looking at
                # written entries (not cards, challenges, etc since they make up a
                # small fraction of the total written text in a story). This
                # works well enough in practice to get decently balanced
                # splits.
                num_words += len(description.split())

        num_scenes = len(scenes)

        total_files += 1
        total_words += num_words
        total_scenes += num_scenes
        total_entries += num_entries

        story_info.append((num_words, num_entries, num_scenes, filename,))

    class Split:
        """
        A class that encapsulates a split and allows for comparisons
        """

        def __init__(self, idx: int, ratio: float):
            self.words = 0
            self.scenes = 0
            self.entries = 0

            self.idx = idx
            self.ratio = ratio
            self.filenames: List[str] = []

        def add(self, words: int, entries: int, scenes: int, filename: str):
            """ Add a file to the split """
            self.words += words
            self.scenes += scenes
            self.entries += entries
            self.filenames.append(filename.replace(data_path, "").lstrip("/"))

        @property
        def weight(self) -> float:
            """ Return the 'weight' of the split """
            return self.words / self.ratio

        def __lt__(self, other: Any) -> bool:
            return (
                self.weight < other.weight
                if isinstance(other, Split)
                else NotImplemented
            )

        def __str__(self) -> str:
            num_files = len(self.filenames)
            return f"Split #{self.idx}: " + ", ".join(
                [
                    f"words={self.words} ({self.words/total_words:.2f})",
                    f"entries={self.entries} ({self.entries/total_entries:.2f})",
                    f"scenes={self.scenes} ({self.scenes/total_scenes:.2f})",
                    f"files={num_files} ({num_files/total_files:.2f})",
                ]
            )

    # Create the priority queue for splits. The heap invariant is the "weight"
    # of the split, but at the start no split has any files, so the order of
    # the heap is arbitrary. Sort based on ratio to make for a deterministic
    # splitting given the same ratios.
    divisor = float(sum(splits))
    split_queue: List[Split] = sorted(
        [Split(idx, split / divisor) for idx, split in enumerate(splits)],
        key=lambda s: s.ratio,
        reverse=True,
    )

    # Do a reverse sort based on the words, entries, and scenes such that we
    # handle the largest stories first. This should give better
    # packing/balancing of splits.
    for words, entries, scenes, filename in sorted(story_info, reverse=True):
        best_split = heapq.heappop(split_queue)
        best_split.add(words, entries, scenes, filename)
        heapq.heappush(split_queue, best_split)

    # Put the splits back into the original order specified for the input
    final_splits = sorted(split_queue, key=lambda s: s.idx)
    for split in final_splits:
        logging.info(split)

    return tuple(s.filenames for s in final_splits)


def perform_split(args):
    """
    Split the dataset according to the passed in args
    """
    splits = split_dataset(
        args.data_directory, (args.train_split, args.validation_split, args.test_split)
    )
    for split, filenames in zip(SPLIT_NAMES, splits):
        with open(
            os.path.join(args.output_directory, f"{split}_filenames.txt"), "wt"
        ) as split_file:
            split_file.write("\n".join(filenames))


def define_split_args(
    sub_parsers: argparse._SubParsersAction,  # pylint:disable=protected-access
):
    """ Define the arguments needed for the split command """
    parser = sub_parsers.add_parser("split", help="Split the dataset")
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
    parser.set_defaults(func=perform_split)
