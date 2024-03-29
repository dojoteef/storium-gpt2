"""
This file contains a number of utility methods for preprocessing stories.
"""
import json
import logging
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto, unique
from itertools import islice, zip_longest
from typing import (Any, Dict, Iterable, List, Optional, Sequence, Tuple,
                    TypeVar, Union)

import torch
from kiwisolver import Constraint, Solver, Variable, strength
from transformers import AutoTokenizer, PreTrainedTokenizer

from data.preprocess import IndexedDict, IndexedSet, Trim


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
        if name == "name":
            # Handle this specially since Enum defines "name" as a string, but
            # we want to use it to extract the field from the data
            name = "name_field"

        return cls(f"<|{name.upper()}|>")

    missing = auto()  # denotes missing information
    separator = auto()  # separator token at the beginning of each Segment
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

    # generic attributes for cards, characters, etc
    name_field = auto()  # cannot call it "name", as Enum defines it as well
    description = auto()

    # Contextual card attributes
    failure_stakes = auto()
    success_stakes = auto()

    # Information denoting entry type
    move = auto()
    establishment = auto()
    addition = auto()
    conclusion = auto()

    # some notions of ordering
    previous = auto()  # can stack, e.g. previous + previous => timestep t-2

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


class Segment(tuple):
    """
    This structure holds a sequence of token ids from the tokenizer vocabulary, along with a
    list of segment ids that modify the entire sequence of token ids.
    """

    # metadata
    segment_ids: Tuple[int, ...]
    separator: Optional[int]
    eos: Optional[int]

    # Layout related variables
    trim: Trim
    constrained: bool
    naive_max_length: int
    preferred_length: int
    length: Optional[Variable]

    # A constant defining the min length of a text segment (it can be shorter
    # if the underlying data is shorter than that constant).
    MIN_LENGTH = 100

    def __new__(
        cls,
        *iterable: Union[Iterable[int], Iterable["Segment"]],
        segment_ids: Iterable[int] = tuple(),
        separator: Optional[int] = None,
        eos: Optional[int] = None,
        preferred_length: int = 0,
        trim: Trim = Trim.end,
    ):
        """
        Create a Segment
        """
        self = super().__new__(cls, *iterable)  # type: ignore

        # Initialize the layout related variables before iterating over the
        # newly created tuple. It must be unconstrained to begin with.
        self.trim = trim
        self.length = None
        self.naive_max_length = -1
        self.preferred_length = preferred_length

        # Since the separator and eos effects the length, we must set it before
        # iterating over the Segment below.
        self.separator = separator
        self.eos = eos

        all_ints = all(isinstance(t, int) for t in self)
        all_segments = all(isinstance(t, Segment) for t in self)
        if not (all_ints or all_segments):
            raise ValueError(
                f"{cls.__name__}() only accepts a homogenous sequence of int or {cls.__name__}"
            )

        self.segment_ids = tuple(segment_ids)
        if not all(isinstance(t, int) for t in segment_ids):
            raise ValueError(f"{cls.__name__}() segment_ids must be int!")

        return self

    @property
    def hard_constraints(self) -> List[Constraint]:
        """
        Return a list of hard constraints defining the length of the segment
        """
        constraints: List[Constraint] = []
        for segment in self:
            if isinstance(segment, Segment):
                constraints.extend(segment.hard_constraints)

        if self.length:
            # Get the underlying length of the data
            length = self.unconstrained_length

            # Cannot be shorter than 1
            constraints.append((self.length >= 1) | strength.required)

            # Cannot be longer than the underlying length
            constraints.append((self.length <= length) | strength.required)

        return constraints

    @property
    def medium_constraints(self) -> List[Constraint]:
        """
        Return a list of constraints defining the length of the segment
        """
        constraints: List[Constraint] = []
        for segment in self:
            if isinstance(segment, Segment):
                constraints.extend(segment.medium_constraints)

        if self.length:
            # Get the underlying length of the data
            length = self.unconstrained_length

            # Resist shrinking below the underlying length
            constraints.append((self.length >= length) | strength.medium)

            # Try to stay close to the underlying length
            constraints.append((self.length == length) | strength.medium)

            # Resist shrinking below the min length a little more strongly
            constraints.append((self.length >= Segment.MIN_LENGTH) | strength.medium)

        return constraints

    @property
    def strong_constraints(self) -> List[Constraint]:
        """
        Return a list of constraints defining the length of the segment
        """
        constraints: List[Constraint] = []
        for segment in self:
            if isinstance(segment, Segment):
                constraints.extend(segment.strong_constraints)

        if self.length:
            # Very strongly try to stay close to the preferred length
            if self.preferred_length:
                constraints.append(
                    (self.length == self.preferred_length) | strength.create(10, 0, 0)
                )
            else:
                # Try to stay close to the min length a little more strongly
                constraints.append(
                    (self.length == Segment.MIN_LENGTH) | strength.strong
                )

        return constraints

    def __len__(self):
        """
        By default length returns the constrained length. To get the full
        length see Segement.unconstrained_length
        """
        unconstrained_length = self.unconstrained_length
        if self.naive_max_length >= 0:
            if not self.preferred_length:
                return min(unconstrained_length, Segment.MIN_LENGTH)

            return unconstrained_length

        if not self.length:
            return unconstrained_length

        return int(self.length.value()) or unconstrained_length

    def __getitem__(self, key):
        """
        Allow __getitem__ to support constraining the underlying sequence
        """
        return self._constrained_sequence[key]

    def __iter__(self):
        """
        Iterate over the constrained Segment
        """
        return iter(self._constrained_sequence)

    @property
    def _constrained_sequence(self):
        """
        Return the constrained segment
        """
        sequence = super().__getitem__(self._constrained_slice)
        if self.separator is not None:
            # If there is a separator it is always the first token
            sequence = (self.separator,) + sequence

            # By virtue of adding the separator, we may need to remove the last
            # element of the sequence, if it would cause the constrained
            # sequence to be too long
            sequence = sequence[: len(self)]

        if self.eos is not None:
            # By virtue of adding eos, we may need to remove the next to last
            # element of the sequence, if it would cause the constrained
            # sequence to be too long
            if len(sequence) + 1 > len(self):
                sequence = sequence[: len(self) - 1]

            # If we have an end of sequence token it is always last
            sequence = sequence + (self.eos,)

        return sequence

    @property
    def _constrained_slice(self) -> slice:
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
        return (
            super().__len__()
            + (0 if self.separator is None else 1)
            + (0 if self.eos is None else 1)
        )

    @property
    def token_segments(self):
        """
        A generator which yields all token ids within the segment recursively,
        along with their associated segment ids, which respects max length.
        """
        if self.length and self.naive_max_length >= 0:
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
    def length_variables(self):
        """
        A generator which yields all the length variables within the segment recursively
        """
        for segment in self:
            if isinstance(segment, Segment):
                yield from segment.length_variables

        if self.length:
            yield self.length

    @property
    def num_tokens(self):
        """
        Get the total number of tokens encapsulated by this Segment
        """
        num_tokens = 0
        for segment in self:
            if not isinstance(segment, Segment):
                # Since we do not mix tokens with nested Segment, we exit early
                # if we do not find a Segment as a minor optimization.
                break

            num_tokens += segment.num_tokens

        if not num_tokens:
            # If we didn't count any tokens, then this could be an empty
            # Segment, or its just make up of tokens without nested Segments.
            num_tokens += self.unconstrained_length

        return num_tokens

    def _mark_constrained(self, constrained: bool, max_length: int = -1):
        """
        Internal method to mark the Segment hierarchy as constrained or not.

        DO NOT CALL THIS DIRECTLY!
        """
        all_ints = False
        for segment in self:
            if not isinstance(segment, Segment):
                # __new__ only accepts a homogenous sequence of int or Segment,
                # so if an element is not a Segment, then this Segment must
                # contain only ints
                all_ints = True
                break

            segment._mark_constrained(  # pylint:disable=protected-access
                constrained, max_length
            )

        # If the segment only contains tokens ids, then it can be constrained.
        self.naive_max_length = max_length
        self.length = Variable() if constrained and all_ints else None

    def _constrain(self, max_length: int):
        """
        Internal method that constrains the Segment to a maximum length using kiwisolver.

        DO NOT CALL THIS DIRECTLY!
        """
        solver = Solver()

        # First add the hard constraints
        for constraint in self.hard_constraints:
            solver.addConstraint(constraint)

        # Want it to be exactly equal to the minimum of the max_length and
        # underlying number of tokens. This makes sure the constraint solver
        # doesn't try to short change and find a solution that uses less than
        # the available length.
        solver.addConstraint(
            sum(self.length_variables) == min(max_length, self.num_tokens)
        )

        # Then add the medium constraints
        for constraint in self.strong_constraints:
            solver.addConstraint(constraint)

        # Finally add the strong constraints
        for constraint in self.strong_constraints:
            solver.addConstraint(constraint)

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


@dataclass
class CharacterInfo:
    """
    The processed character info
    """

    summary: Segment
    character_id: str
    checksum: int

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
    checksum: int
    text: Segment
    summary: Segment


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


def extract_string(
    field: str, mapping: Dict[str, Any], default: str = SpecialToken.missing.value
) -> str:
    """
    Extract the given string field, accounting for the potential that it is
    specified as None
    """
    return mapping.get(field, default) or default


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
        naive_layout: bool = False,
    ):
        """
        NOTE: The default max_length parameter was chosen to support GPT-2.
        """
        self.tokenizer = tokenizer

        self.history = history
        self.character_history = character_history

        self.max_length = max_length
        self.preferred_entry_length = preferred_entry_length
        self.naive_layout = naive_layout

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
        special_token: Optional[SpecialToken] = None,
        separator_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        preferred_length: int = 0,
        trim: Trim = Trim.end,
    ) -> Segment:
        """
        After encoding with the tokenizer, this creates, create a Segment and
        assign the special_token if specified.
        """
        return Segment(
            self.encode(string_or_list),
            separator=self.tokenizer.convert_tokens_to_ids(SpecialToken.separator)
            if separator_token_id is None
            else separator_token_id,
            eos=eos_token_id,
            segment_ids=[self.tokenizer.convert_tokens_to_ids(special_token)]
            if special_token
            else tuple(),
            preferred_length=preferred_length,
            trim=trim,
        )

    def checksum_character(self, character: Dict[str, Any], character_id: str) -> int:
        """
        Compute a checksum of a character
        """
        checksum = zlib.adler32(character_id.encode("utf-8"))
        for field in ("name", "description"):
            checksum = zlib.adler32(
                extract_string(field, character).encode("utf-8"), checksum
            )

        return checksum

    def summarize_character(self, character: Dict[str, Any]) -> Segment:
        """
        Create the summary for a character
        """
        return Segment(
            iter(
                self.encode_special(
                    extract_string(field, character),
                    SpecialToken.from_string(field),
                    separator_token_id=self.tokenizer.bos_token_id
                    if field == "name"
                    else None,
                )
                for field in ("name", "description")
            ),
            segment_ids=[
                self.tokenizer.convert_tokens_to_ids(SpecialToken.character),
            ],
        )

    def checksum_card(self, card: Optional[Dict[str, Any]], checksum: int = 1) -> int:
        """
        Checksum the card.
        """
        if not card:
            return checksum

        for field in ("name", "description", "success_stakes", "failure_stakes"):
            checksum = zlib.adler32(
                extract_string(field, card).encode("utf-8"), checksum
            )

        return checksum

    def summarize_card(self, card: Optional[Dict[str, Any]]) -> Segment:
        """
        Create the summary of a card.

        If it's a challenge card, then it'll have "success_stakes" and
        "failure_stakes" as well.
        """
        if not card:
            return Segment()

        return Segment(
            iter(
                self.encode_special(
                    extract_string(field, card),
                    SpecialToken.from_string(field),
                )
                for field in ("name", "description", "success_stakes", "failure_stakes")
                if card.get(field)
            ),
            segment_ids=tuple(
                self.tokenizer.convert_tokens_to_ids(
                    (SpecialToken.from_string(card["namespace"]),)
                ),
            ),
        )

    def checksum_cards(self, cards: List[Dict[str, Any]], checksum: int = 1) -> int:
        """
        Create the summary of a card
        """
        for card in cards:
            checksum = self.checksum_card(card, checksum)

        return checksum

    def summarize_cards(self, cards: List[Dict[str, Any]]) -> Segment:
        """
        Create the summary of a card
        """
        return Segment(iter(self.summarize_card(card) for card in cards))

    def checksum_entry(self, entry: Dict[str, Any], entry_id: str) -> int:
        """
        Compute a checksum of an entry
        """
        checksum = zlib.adler32(entry_id.encode("utf-8"))
        entry_type = entry["format"]
        if entry_type == "move":
            checksum = self.checksum_card(entry.get("target_challenge_card"), checksum)
            checksum = self.checksum_cards(
                entry.get("cards_played_on_challenge", []), checksum
            )
        elif entry_type == "establishment":
            checksum = self.checksum_card(entry.get("place_card"), checksum)
        elif entry_type == "addition":
            checksum = self.checksum_cards(entry.get("challenge_cards", []), checksum)

        return zlib.adler32(
            extract_string("description", entry, "").encode("utf-8"), checksum
        )

    def summarize_entry(self, entry: Dict[str, Any]) -> Segment:
        """
        Create the summary of an entry
        """
        summary = []
        entry_type = entry["format"]
        if entry_type == "move":
            challenge = self.summarize_card(entry.get("target_challenge_card"))
            if challenge:
                summary.append(challenge)

            cards = self.summarize_cards(entry.get("cards_played_on_challenge", []))
            if cards:
                summary.append(cards)
        elif entry_type == "establishment":
            place = self.summarize_card(entry.get("place_card"))
            if place:
                summary.append(place)
        elif entry_type == "addition":
            cards = self.summarize_cards(entry.get("challenge_cards", []))
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
        self,
        entry: Dict[str, Any],
        establishment_id: str,
        checksum: int,
        add_eos: bool = True,
        force: bool = False,
    ) -> Optional[EntryInfo]:
        """
        Process a character entry
        """
        text = extract_string("description", entry, "")
        if not text and not force and entry.get("format") != "establishment":
            # Only modeling moves with written text, though make a special
            # exception for establishment entries. While they are currently
            # required to have text, it seems at some point there were games that
            # didn't have any text for the establishment entry, though it would still
            # have place cards.
            return None

        encoded_text = self.encode_special(
            text,
            SpecialToken.from_string(entry["format"]),
            preferred_length=self.preferred_entry_length,
            eos_token_id=self.tokenizer.eos_token_id if add_eos else None,
        )
        summary = self.summarize_entry(entry)
        if not summary:
            summary = self.encode_special(
                text,
                SpecialToken.from_string(entry["format"]),
                trim=Trim.start,  # Treat the end of the entry text as a summary
            )

        return EntryInfo(
            checksum=checksum,
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

    def process_story(
        self, story: Dict[str, Any], processed: Optional[ProcessedStory] = None
    ) -> Optional[ProcessedStory]:
        """
        Summarize a story, potentially based off of a previously processed story.

        Returns a ProcessedStory
        """
        scenes = story.get("scenes")
        characters = story.get("characters")
        if not scenes or not characters or not isinstance(scenes, Sequence):
            # Return whatever was previously processed (if anything)
            return processed

        character_list = [
            # Treat narrator as a character who is always present without a summary
            (
                "narrator",
                CharacterInfo(
                    checksum=0,
                    entry_ids=IndexedSet(),
                    character_id="narrator",
                    summary=Segment(),
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

            # Get the previously processed character if any
            character_info = (
                processed.characters.get(character_id, None) if processed else None
            )

            # Compute the checksum for the character
            checksum = self.checksum_character(character, character_id)
            if not character_info or character_info.checksum != checksum:
                # Haven't processed this character before, so process it now
                character_info = CharacterInfo(
                    checksum=checksum,
                    entry_ids=IndexedSet(),
                    character_id=character_id,
                    summary=self.summarize_character(character),
                )

            character_list.append(
                (
                    character_id,
                    character_info,
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

                # Compute the checksum for the entry
                checksum = self.checksum_entry(entry, entry_id)

                # Get the previously processed entry if any
                entry_info = (
                    processed.entries.get(entry_id, None) if processed else None
                )
                if not entry_info or entry_info.checksum != checksum:
                    # Haven't processed this entry before, so process it now
                    entry_info = self.process_entry(
                        entry,
                        establishment_list[-1][0] if establishment_list else entry_id,
                        checksum,
                    )

                if not entry_info:
                    continue

                entry_list.append((entry_id, entry_info))
                entry_format = entry.get("format")
                if entry_format == "establishment":
                    establishment_list.append((entry_id, entry_info))

                # See https://github.com/PyCQA/pylint/issues/3129
                character_info = (
                    all_characters[  # pylint:disable=unsubscriptable-object
                        entry["role"]
                    ]
                )
                character_info.entry_ids.insert(entry_id)

        return ProcessedStory(
            game_id=story["game_pid"],
            entries=IndexedDict(entry_list),
            characters=all_characters,
            establishment_entries=IndexedDict(establishment_list),
        )

    def get_entry_summary(
        self, story: ProcessedStory, entry_info: EntryInfo
    ) -> Segment:
        """
        Extracts the context for a given entry
        """
        game_id = story.game_id
        entry_id = entry_info.entry_id
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

        try:
            # Get the index of the entry for the character
            end_index = max(0, character_info.entry_ids.index(entry_id) - 1)
        except ValueError:
            # If the entry cannot be found, then it is a new entry that hasn't
            # been put into the character's list of entries, so use the index
            # of the last character entry
            end_index = len(character_info.entry_ids) - 1

        start_index = max(0, end_index - self.character_history)
        prev_entry_ids = IndexedSet(
            character_info.entry_ids[idx] for idx in range(start_index, end_index)
        )

        try:
            # Get the index of the entry for the character
            end_index = max(0, story.entries.index(entry_id) - 1)
        except ValueError:
            # If the entry cannot be found, then it is a new entry that hasn't
            # been put into the story's list of entries, so use the index
            # of the last story entry
            end_index = len(story.entries) - 1

        start_index = max(0, end_index - self.history)
        for entry_index in range(start_index, end_index):
            prev_entry_ids.insert(story.entries.indices[entry_index])

        for idx, prev_entry_id in enumerate(prev_entry_ids, 1):
            # Get the summary of the previous entries from the same character and
            # mark them as such
            prev_entry_info = story.entries[prev_entry_id]
            if prev_entry_info.entry_id == entry_info.establishment_id:
                continue

            character_token = (
                SpecialToken.same_character
                if prev_entry_info.character_id == character_id
                else SpecialToken.diff_character
            )
            summary.append(
                Segment(
                    prev_entry_info.summary,
                    segment_ids=tuple(
                        self.tokenizer.convert_tokens_to_ids(
                            idx * (SpecialToken.previous,) + (character_token,)
                        )
                    ),
                )
            )

        if entry_info.summary != entry_info.text:
            entry_summary = entry_info.summary

            if prev_entry_ids:
                # If we used any history, then mark this entry as written by the
                # same character
                entry_summary = Segment(
                    entry_summary,
                    segment_ids=[
                        self.tokenizer.convert_tokens_to_ids(
                            SpecialToken.same_character
                        )
                    ],
                )
            summary.append(entry_summary)

        return Segment(summary)

    def get_move(self, story: ProcessedStory, entry_info: EntryInfo) -> Segment:
        """
        Extracts the context for a given entry
        """
        if entry_info.character_id == "narrator":
            # Only characters can do a normal "move"
            return Segment()

        return Segment((self.get_entry_summary(story, entry_info), entry_info.text))
