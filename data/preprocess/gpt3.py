"""
This file contains a number of utility methods for preprocessing stories.
"""
import json
import logging
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, unique
from typing import (Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple,
                    Union)

from kiwisolver import Constraint, Solver, Variable, strength
from transformers import PreTrainedTokenizer

from data.preprocess import IndexedDict, IndexedSet, Trim


@unique
class SpecialToken(str, Enum):
    """
    An enumeration of special tokens
    """

    @classmethod
    def from_string(cls, name: str):
        """
        Get the associated SpecialToken from the passed in string
        """
        return cls.__members__[name]

    missing = "*MISSING*"  # denotes missing information
    character = "#CHARACTER — "  # a character's biography

    # All the possible card types from <CardNamespace> in the Storium export
    # format: https://storium.com/help/export/json/0.9.2
    chartype = "##NATURE — "
    goal = "##GOAL — "
    person = "##CHARACTER — "
    place = "##LOCATION — "
    thing = "##ASSET — "
    strength = "##STRENGTH — "
    weakness = "##WEAKNESS — "
    obstacle = "##CHALLENGE — "
    subplot = "##SUBPLOT — "

    # Contextual card attributes
    failure_stakes = "###ON FAILURE\n"
    success_stakes = "###ON SUCCESS\n"

    # Information denoting entry type
    move = "#SCENE ENTRY\n"
    establishment = "#SCENE ESTABLISHMENT\n"
    addition = "#SCENE CONTINUATION\n"
    conclusion = "#SCENE CONCLUSION\n"

    move_summary = "#SCENE ENTRY SUMMARY\n"
    establishment_summary = "#SCENE ESTABLISHMENT SUMMARY\n"
    addition_summary = "#SCENE CONTINUATION SUMMARY\n"
    conclusion_summary = "#SCENE CONCLUSION SUMMARY\n"

    # The scene entry we want to learn to generate. Use a short string to appease openai's
    # fine tuning data prepration tool
    current_move = "#AND NOW\n"

    def __str__(self):
        """
        Override the default string method to return the enumeration value,
        which is a string
        """
        return self.value


class ConstraintSet(List["Segment"]):
    """
    Set of Segments that are treated as all or nothing in terms of constraints. If any of
    the Segments have a length of zero in the ConstraintSet, then all the length variables
    are forced to be zero.
    """

    @property
    def empty(self):
        """Is the constraint set empty?"""
        return all(
            s.length.value() == 0 or s.fixed_length
            for s in self
            if s.length is not None
        )

    @property
    def violated(self):
        """Do any length variables have a value of 0"""
        return any(s.length.value() == 0 for s in self if s.length is not None)

    def difference(self, *other: Iterable["ConstraintSet"]) -> "ConstraintSet":
        """
        Compute a ConstraintSet that removes any elements from the list of other ConstraintSets
        """
        return ConstraintSet(s for s in self if not any(s in o for o in other))


class Segment(Tuple[Union[int, "Segment"]]):
    """
    This structure holds a sequence of token ids from the tokenizer vocabulary, along with a
    list of segment ids that modify the entire sequence of token ids.
    """

    # Constants
    # Layout related variables
    trim: Trim
    leaf: bool
    atomic: bool
    fixed_length: bool
    preferred_length: int
    preferred_strength: int
    length: Optional[Variable]
    ellipsis: Tuple[int, ...]
    hard_constraints: List[Constraint]
    strong_constraints: List[Constraint]
    medium_constraints: List[Constraint]
    full_constraint_set: ConstraintSet
    disjoint_constraint_set: ConstraintSet

    # A constant defining the min length of a text segment (it can be shorter
    # if the underlying data is shorter than that constant).
    MIN_LENGTH = 100

    def __new__(
        cls,
        *iterable: Union[Iterable[int], Iterable["Segment"]],
        trim: Trim = Trim.middle,
        fixed_length: bool = False,
        preferred_length: int = 0,
        preferred_strength: int = 1,
        atomic: bool = False,
        ellipsis: Sequence[int] = tuple(),
    ):
        """
        Create a Segment
        """
        self = super().__new__(cls, *iterable)  # type: ignore

        all_ints = all(isinstance(t, int) for t in super(cls, self).__iter__())
        all_segments = all(isinstance(t, Segment) for t in super(cls, self).__iter__())
        if not (all_ints or all_segments):
            raise ValueError(
                f"{cls.__name__}() only accepts a homogenous sequence of int or {cls.__name__}"
            )

        self.trim = trim
        self.ellipsis = tuple(ellipsis)
        self.length = None
        self.hard_constraints = []
        self.strong_constraints = []
        self.medium_constraints = []
        self.leaf = all_ints
        self.fixed_length = fixed_length
        self.atomic = atomic
        self.preferred_length = preferred_length
        self.preferred_strength = preferred_strength

        # Compute the constraint sets
        if self.atomic_container:
            self.full_constraint_set = ConstraintSet(
                (s for s in self.segments if s.leaf)
            )
            self.disjoint_constraint_set = ConstraintSet(
                self.full_constraint_set.difference(
                    *(
                        child.full_constraint_set
                        for child in super(cls, self).__iter__()
                    )
                ),
            )
        else:
            self.full_constraint_set = ConstraintSet()
            self.disjoint_constraint_set = ConstraintSet()

        return self

    def clone(
        self,
        *,
        trim: Optional[Trim] = None,
        fixed_length: Optional[bool] = None,
        preferred_length: Optional[int] = None,
        preferred_strength: Optional[int] = None,
        atomic: Optional[bool] = None,
        ellipsis: Optional[Sequence[int]] = None,
        max_length: Optional[int] = None,
    ) -> "Segment":
        """
        Clone the segment, overwriting the passed in attributes
        """

        def nested_clone(segment):
            """Method to deep clone the segment with updated parameters"""

            def choose(key: str, value: Optional[Any] = None):
                """Use the override if specified"""
                return getattr(segment, key) if value is None else value

            kwargs = {
                "trim": choose("trim", trim),
                "ellipsis": choose("ellipsis", ellipsis),
                "fixed_length": choose("fixed_length", fixed_length),
                "preferred_length": choose("preferred_length", preferred_length),
                "atomic": choose("atomic", atomic),
                "preferred_strength": choose("preferred_strength", preferred_strength),
            }

            iterator = super(Segment, segment).__iter__()
            return type(segment)(
                (
                    (
                        x
                        for i, x in enumerate(iterator)
                        if max_length is None or i < max_length
                    )
                    if segment.leaf
                    else (
                        nested_clone(segment_or_token) for segment_or_token in iterator
                    )
                ),
                **kwargs,
            )

        return nested_clone(self)

    @property
    def unconstrained_length(self):
        """
        The length of the segment when unconstrained
        """
        return super().__len__()

    def __len__(self):
        """
        By default length returns the constrained length. To get the full
        length see Segment.unconstrained_length
        """
        if not self.length:
            return self.unconstrained_length

        return int(self.length.value())

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
        parts = []
        start, end, ellipsis = self._trim()
        if start:
            parts.append(super().__getitem__(start))

        if ellipsis:
            parts.append(self.ellipsis)

        if end:
            parts.append(super().__getitem__(end))

        return tuple(x for part in parts for x in part)

    def _trim(self) -> Tuple[Optional[slice], Optional[slice], bool]:
        """
        Trim the Segment if needed by returning slices of the Segment
        """
        length = len(self)
        if length == self.unconstrained_length:
            return slice(None), None, False

        # Extra length to account for the ellipsis
        pad = len(self.ellipsis)

        # If it's too small for the ellipsis, remove it entirely
        if length <= pad:
            return slice(0, 0), None, length == pad

        if self.trim is Trim.end:
            return slice(length - pad), None, True

        if self.trim is Trim.start:
            return None, slice(self.unconstrained_length + pad - length, None), True

        if self.trim is Trim.middle:
            midpoint = self.unconstrained_length // 2
            extra = self.unconstrained_length - length
            idx = midpoint - extra // 2
            return (
                slice(max(0, idx - pad)),
                slice(self.unconstrained_length - idx, None),
                True,
            )

        raise RuntimeError("Unknown trim type!")

    @property
    def depth(self):
        """How deep are the nested Segments within this segmented"""

        def _depth(segment, depth):
            max_depth = depth
            for segment in super().__iter__():
                if isinstance(segment, Segment):
                    max_depth = max(max_depth, _depth(segment, depth + 1))

            return max_depth

        if self.container:
            return _depth(self, 1)

        return 1

    @property
    def segments(self):
        """
        Generator that yields the current segment and any nested segments in post order
        """
        for segment in super().__iter__():
            if isinstance(segment, Segment):
                yield from segment.segments

        yield self

    @property
    def tokens(self):
        """
        A generator which yields all token ids within the segment recursively
        """
        for segment_or_token in self:
            if isinstance(segment_or_token, Segment):
                yield from segment_or_token.tokens
            else:
                yield segment_or_token

    @property
    def length_variables(self):
        """
        A generator which yields all the length variables within the segment recursively
        """
        for segment in self.segments:
            if segment.length:
                yield segment.length

    @property
    def unconstrained_num_tokens(self):
        """
        Get the total number of tokens encapsulated by this Segment
        """
        return sum(s.unconstrained_length for s in self.segments if s.leaf)

    @property
    def container(self):
        """Does this Segment container other segments?"""
        return not self.leaf

    @property
    def atomic_container(self):
        """Is this a Segment atomic and a container?"""
        return self.atomic and self.container

    @property
    def full_constraint_sets(self):
        """
        Generator that yields full constraint sets for nested Segments using post order
        traversal. This makes it such that we remove the leaves first when trying to resolve
        constraint violations.
        """
        if not self.container:
            return

        for segment in super().__iter__():
            if isinstance(segment, Segment):
                yield from segment.full_constraint_sets

        yield self.full_constraint_set

    @property
    def disjoint_constraint_sets(self):
        """
        Generator that yields disjoint constraint sets for nested Segments using post order
        traversal. This makes it such that we remove the leaves first when trying to resolve
        constraint violations.
        """
        if not self.container:
            return

        for segment in super().__iter__():
            if isinstance(segment, Segment):
                yield from segment.disjoint_constraint_sets

        yield self.disjoint_constraint_set

    def _mark_constrained(self, constrained: bool):
        """
        Internal method to mark the Segment hierarchy as constrained or not.

        DO NOT CALL THIS DIRECTLY!
        """
        for segment in self.segments:
            if not segment.leaf:
                continue

            segment.hard_constraints = []
            segment.medium_constraints = []
            segment.strong_constraints = []

            if not constrained:
                segment.length = None
                continue

            segment.length = Variable()

            if segment.fixed_length:
                # Make sure it is equal to the unconstrained length
                segment.hard_constraints.append(
                    (segment.length == segment.unconstrained_length) | strength.required
                )
            else:
                ######################
                # Required constraints
                ######################

                # Length must be non-negative
                segment.hard_constraints.append(
                    (segment.length >= 0) | strength.required
                )

                # Cannot be longer than the underlying length
                segment.hard_constraints.append(
                    (segment.length <= segment.unconstrained_length) | strength.required
                )

                ######################
                # Strong constraints
                ######################
                depth_factor = 10 * segment.depth

                # Very strongly try to stay close to the preferred length
                if segment.preferred_length:
                    segment.strong_constraints.append(
                        (segment.length == segment.preferred_length)
                        | strength.create(
                            10 * segment.preferred_strength, depth_factor, 0
                        )
                    )
                else:
                    # Try to stay close to the min length a little more strongly
                    segment.strong_constraints.append(
                        (segment.length == Segment.MIN_LENGTH)
                        | strength.create(1, depth_factor, 0)
                    )

                ######################
                # Medium constraints
                ######################
                medium_strength = strength.create(0, depth_factor, 0)

                # Resist shrinking below the underlying length
                segment.medium_constraints.append(
                    (segment.length >= segment.unconstrained_length) | medium_strength
                )

                # Try to stay close to the underlying length
                segment.medium_constraints.append(
                    (segment.length == segment.unconstrained_length) | medium_strength
                )

                # Resist shrinking below the min length a little more strongly
                segment.medium_constraints.append(
                    (segment.length >= Segment.MIN_LENGTH) | medium_strength
                )

    def _constrain(self, max_length: int):
        """
        Internal method that constrains the Segment to a maximum length using kiwisolver.

        DO NOT CALL THIS DIRECTLY!
        """
        solver = Solver()

        # First add the hard constraints
        for segment in self.segments:
            for constraint in segment.hard_constraints:
                solver.addConstraint(constraint)

        # Want it to be exactly equal to the minimum of the max_length and
        # underlying number of tokens. This makes sure the constraint solver
        # doesn't try to short change and find a solution that uses less than
        # the available length.
        solver.addConstraint(
            (
                sum(self.length_variables)
                == min(max_length, self.unconstrained_num_tokens)
            )
            | strength.required
        )

        # Then add the strong constraints
        for segment in self.segments:
            for constraint in segment.strong_constraints:
                solver.addConstraint(constraint)

        # Finally add the medium constraints
        for segment in self.segments:
            for constraint in segment.medium_constraints:
                solver.addConstraint(constraint)

        # Now update the length variables
        solver.updateVariables()

        removed_segments: Set[int] = set()
        removed_constraint_sets: Set[int] = set()

        def collect_violated(segment):
            """Return any empty or violated constraint sets"""
            return [
                cs
                for cs in segment.disjoint_constraint_sets
                if cs.violated and id(cs) not in removed_constraint_sets
            ]

        def collect_empty(segment: Segment):
            """Return any empty constraint sets for the passed in Segment"""
            return [
                cs
                for cs in segment.disjoint_constraint_sets
                if cs.empty and id(cs) not in removed_constraint_sets
            ]

        def removeConstraints(segment: Segment):
            """Remove all length constraints"""
            if id(segment.length) in removed_segments:
                return

            for constraint in segment.hard_constraints:
                solver.removeConstraint(constraint)

            for constraint in segment.strong_constraints:
                solver.removeConstraint(constraint)

            for constraint in segment.medium_constraints:
                solver.removeConstraint(constraint)

            # Then force segment to have a zero length
            # solver.addConstraint((segment.length == 0) | strength.required)
            solver.addConstraint((segment.length >= 0) | strength.required)
            solver.addConstraint((segment.length == 0) | strength.strong)

            # Update the length variables after removing constraints
            solver.updateVariables()

            # And track that we removed this segment
            removed_segments.add(id(segment.length))

        def removeConstraintSet(constraint_set: ConstraintSet):
            """Remove all segments in the ConstraintSet"""
            removed_constraint_sets.add(id(constraint_set))

            for segment in constraint_set:
                removeConstraints(segment)

                for empty in collect_empty(segment):
                    removeConstraintSet(empty)

        # Now remove any constraint sets that have been violated one by one until the
        # solver succeeds
        still_violated = collect_violated(self)
        while still_violated:
            # Remove the violated constraint set
            removeConstraintSet(still_violated.pop(0))

            # Collect any remaining constraint sets that are still violated
            still_violated = collect_violated(self)

    @contextmanager
    def constraint(self, max_length: int):
        """
        A context manager that constrains the Segment, performs the necessary
        operation then unconstrains the Segment.
        """
        self._mark_constrained(True)
        self._constrain(max_length)
        yield
        self._mark_constrained(False)

    def flatten(self) -> Tuple[int, ...]:
        """
        Recursively flatten the segment, returning a tuple of tokens
        """
        return tuple(self.tokens)

    def aslist(self) -> List[int]:
        """
        Recursively flatten the segment and return it as a list of tokens
        """
        return list(self.flatten())


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
        max_entry_length: int = 256,
        preferred_entry_length: int = 256,
    ):
        """
        Args:
            tokenizer - which tokenizer to use
            history - # of scene entries to use as history
            character_history - # of scene entries from the current character to use as history
            max_length - max number of tokens for a preprocessed story
            max_entry_length - the max length of the final entry
            preferred_entry_length - the desired length of each entry

        NOTE: The default max_length parameter was chosen to support GPT-2.
        """
        self.tokenizer = tokenizer

        self.history = history
        self.character_history = character_history

        self.max_length = max_length
        self.max_entry_length = max_entry_length
        self.preferred_entry_length = preferred_entry_length

    def encode_token(self, token: str) -> List[int]:
        """
        Encode the token
        """
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer._tokenize(token))

    def encode_special_token(self, token: Union[str, SpecialToken]) -> List[int]:
        """
        Encode the special token
        """
        if not isinstance(token, SpecialToken):
            token = SpecialToken.from_string(token)

        return self.encode_token(token)

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
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

        max_length = max_length or self.max_length
        tokens = self.tokenizer.encode(text, max_length=max_length)
        logger.setLevel(log_level)
        return tokens

    def encode_text(
        self,
        string_or_list: Union[str, List[str]],
        header: Optional[SpecialToken] = None,
        separator: Optional[str] = "\n\n",
        preferred_length: int = 0,
        trim: Trim = Trim.middle,
        fixed_length: bool = False,
    ) -> Segment:
        """
        After encoding with the tokenizer, this creates, create a Segment and
        assign the special_token if specified.
        """
        sections = []
        if header:
            sections.append(
                Segment(
                    self.encode_special_token(header),
                    fixed_length=True,
                )
            )

        text = (
            " ".join(string_or_list)
            if isinstance(string_or_list, list)
            else string_or_list
        )
        ellipsis = "\n...\n" if trim == Trim.middle and "\n" in text else "..."
        sections.append(
            Segment(
                self.encode(text),
                preferred_length=preferred_length,
                trim=trim,
                ellipsis=self.encode_token(ellipsis),
                fixed_length=fixed_length,
            )
        )

        if separator:
            sections.append(
                Segment(
                    self.encode_token(separator),
                    fixed_length=True,
                )
            )

        return Segment(sections)

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
        sections = [
            Segment(
                self.encode_special_token(SpecialToken.character),
                fixed_length=True,
            )
        ]
        sections.append(
            self.encode_text(
                extract_string("name", character),
                fixed_length=True,
                separator="\n",
            )
        )
        sections.append(self.encode_text(extract_string("description", character)))

        return Segment(sections, atomic=True)

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

        sections = [
            Segment(
                self.encode_special_token(card["namespace"]),
                fixed_length=True,
            )
        ]
        if card.get("name"):
            sections.append(
                self.encode_text(
                    extract_string("name", card),
                    fixed_length=True,
                    separator="\n",
                )
            )

        if card.get("description"):
            sections.append(
                self.encode_text(
                    extract_string("description", card),
                    separator="\n",
                )
            )

        for field in ("success_stakes", "failure_stakes"):
            if card.get(field):
                sections.append(
                    self.encode_text(
                        extract_string(field, card),
                        header=SpecialToken.from_string(field),
                        separator="\n",
                    )
                )

        # Add a final separator
        sections.append(
            Segment(
                self.encode_token("\n"),
                fixed_length=True,
            )
        )

        return Segment(sections, atomic=True)

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
        return Segment(iter(self.summarize_card(card) for card in cards), atomic=True)

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

        if not summary:
            return Segment()

        header = Segment(
            self.encode_special_token(entry_type + "_summary"),
            fixed_length=True,
        )
        return Segment([header] + summary, atomic=True)

    def process_entry(
        self,
        entry: Dict[str, Any],
        establishment_id: str,
        checksum: int,
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

        encoded_text = self.encode_text(
            text or " ", header=SpecialToken.from_string(entry["format"])
        )
        summary = self.summarize_entry(entry)
        if not summary:
            summary = self.encode_text(
                text,
                header=SpecialToken.from_string(entry["format"]),
                trim=Trim.middle,  # Treat a trimmed version of the text as a summary
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

    def collect_history(
        self,
        story: ProcessedStory,
        character_info: CharacterInfo,
        entry_info: EntryInfo,
    ) -> List[EntryInfo]:
        """
        Collect the list of entries to use as history for the passed in entry
        """
        entry_id = entry_info.entry_id
        history: IndexedSet[str] = IndexedSet()
        if entry_id != entry_info.establishment_id:
            establishment_info = story.establishment_entries.get(
                entry_info.establishment_id
            )
            if not establishment_info:
                raise KeyError(
                    f"""
                    Cannot find establishment entry {entry_info.establishment_id} for story {story.game_id}!
                    """
                )

            history.insert(entry_info.establishment_id)

        try:
            # Get the index of the entry for the character
            end_index = max(0, character_info.entry_ids.index(entry_id) - 1)
        except ValueError:
            # If the entry cannot be found, then it is a new entry that hasn't
            # been put into the character's list of entries, so use the index
            # of the last character entry
            end_index = len(character_info.entry_ids) - 1

        start_index = max(0, end_index - self.character_history)
        for idx in range(start_index, end_index):
            history.insert(character_info.entry_ids[idx])

        try:
            # Get the index of the previous entry
            end_index = max(0, story.entries.index(entry_id) - 1)
        except ValueError:
            # If the entry cannot be found, then it is a new entry that hasn't
            # been put into the story's list of entries, so use the index
            # of the last story entry
            end_index = len(story.entries) - 1

        start_index = max(0, end_index - self.history)
        for entry_index in range(start_index, end_index):
            history.insert(story.entries.indices[entry_index])

        return [story.entries[entry_id] for entry_id in history]

    def compile_entries(
        self, entries: Sequence[EntryInfo], add_suffix: bool = True
    ) -> Segment:
        """
        Compile the list of entries into a single Segment
        """
        compiled = []
        for i, entry_info in enumerate(entries, 1):
            compiled.append(entry_info.summary.clone())

            if entry_info.summary != entry_info.text:
                if i < len(entries):
                    compiled.append(
                        entry_info.text.clone(
                            preferred_strength=i,
                            preferred_length=self.preferred_entry_length,
                        )
                    )
                else:
                    # Need to specially format the entry we want to model for completions
                    # 1. It must begin with a unique prefix (actually becomes the suffix of
                    # the GPT-3 prompt)
                    final_entry = []
                    final_entry.append(
                        Segment(
                            self.encode_special_token(SpecialToken.current_move),
                            fixed_length=True,
                        )
                    )

                    # 2. Add a space after the prefix since GPT-3 really wants completions
                    # to begin with a space
                    final_entry.append(
                        Segment(
                            self.encode_token(" "),
                            fixed_length=True,
                        )
                    )

                    # There might not be any text for a completion
                    if entry_info.text[1]:
                        # 3. We want to trim the end of the text to a maximum length without
                        # any ellipsis (no use in modeling more completion tokens than our max
                        # generation length at inference)
                        final_entry.append(
                            entry_info.text[1].clone(
                                trim=Trim.end,
                                preferred_strength=i,
                                preferred_length=self.preferred_entry_length,
                                max_length=self.max_entry_length,
                                ellipsis=tuple(),
                            )
                        )

                        # Only include the suffix if specified and there is text. Otherwise
                        # we are encoding a completion
                        if add_suffix:
                            # 4. It must end with a unique suffix
                            final_entry.append(
                                Segment(
                                    self.encode_token("$$$"),
                                    fixed_length=True,
                                )
                            )

                    compiled.append(Segment(final_entry, atomic=True))

        return Segment(compiled)

    def get_move(
        self, story: ProcessedStory, entry_info: EntryInfo, add_suffix: bool = False
    ) -> Segment:
        """
        Extracts the context for a given entry
        """
        if entry_info.character_id == "narrator":
            # Only characters can do a normal "move"
            return Segment()

        if entry_info.summary == entry_info.text:
            # Only consider entries that have a summary (e.g. cards played on a challenge)
            return Segment()

        character_id = entry_info.character_id
        character_info = story.characters.get(character_id)
        if not character_info:
            raise KeyError(
                f"Cannot find character {character_id} for story {story.game_id}!"
            )

        history = self.collect_history(story, character_info, entry_info)
        entries = history + [entry_info]
        return Segment(
            (
                # Allow equal strength to character info as the main entry we are modeling
                character_info.summary.clone(
                    preferred_strength=len(entries),
                    preferred_length=self.preferred_entry_length,
                ),
                self.compile_entries(entries, add_suffix=add_suffix),
            )
        )

    def decode(self, segment: Segment) -> str:
        """
        Decode the passed in segment as a regular string
        """
        return self.tokenizer.decode(
            segment.flatten(), clean_up_tokenization_spaces=False
        )
