"""
This file contains a number of utility methods for preprocessing stories.


NOTE: The functionality is currently based on the preprocessing approach taken
by Shufan in prepare_dataset_two.py. I'm really not a fan of some of the
decisions taken, but I've tried to make sure the code here mirrors what was
previously written so the results will be consistent.
"""
import os
import glob
import json
import heapq
import logging

from bisect import bisect_left
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple
from itertools import chain

from pydantic import BaseModel


###############################################################################
# IDEA:
# Make all the summarize_* methods take in a dropout probability, such that
# they randomly dropout a component as SpecialTokens.missing, but the loss is
# still calculated against the full sequence, such that it can learn to fill in
# missing data.
#
# This, for example, could be used as a simple technique for predicting a card
# to play for the "wild" cards
###############################################################################


class SpecialTokens(str, Enum):
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

    card = auto()
    entry = auto()
    missing = auto()
    padding = auto()
    challenge = auto()
    character = auto()
    failure_stakes = auto()
    success_stakes = auto()

    @classmethod
    def as_dict(cls):
        """
        Return a dictionary of enumeration name to its string representation
        """
        return {k: str(v) for k, v in cls.__members__.items()}

    def __str__(self):
        """
        Override the default string method to return the enumeration value,
        which is a string
        """
        return self.value


class CharacterInfo(BaseModel):
    """
    The processed character info
    """

    summary: str
    character_id: str

    # This is a sorted list of entry ids written by the character to
    # allow easily looking up the previous entries for the character
    entry_ids: List[str]

    def index(self, entry_id):
        """
        Return the index of the specified entry id in the list.

        Locates the leftmost value exactly equal to entry_id. Based on the
        index method from: https://docs.python.org/3/library/bisect.html
        """
        idx = bisect_left(self.entry_ids, entry_id)
        if idx != len(self.entry_ids) and self.entry_ids[idx] == entry_id:
            return idx

        raise ValueError(f"{entry_id} not found!")


class EntryInfo(BaseModel):
    """
    The processed entry info
    """

    text: str
    entry_id: str

    summary: str
    character_id: str


class CardInfo(BaseModel):
    """
    The processed card info
    """

    text: str
    entry_id: str

    summary: str
    character_id: str


class ProcessedStory(BaseModel):
    """
    This defines the structure of a story after processing
    """

    # A mapping of entry id to of entry info
    entries: Dict[str, EntryInfo]

    # A mapping of character id to character info
    characters: Dict[str, CharacterInfo]


def extract_string(field: str, mapping: Dict[str, Any]) -> str:
    """
    Extract the given string field, accounting for the potential that it is
    specified as None
    """
    return mapping.get(field, SpecialTokens.missing) or SpecialTokens.missing


def summarize_character(character: Dict[str, Any], max_length: int = 100) -> str:
    """
    Create the summary for a character

    NOTE: The default max_length parameter was chosen empirically to make the
    context size less than half the max length supported by GPT-2 (1024).
    """
    strings: List[str] = [SpecialTokens.character]
    if character:
        strings.extend(extract_string("name", character).split())
        strings.extend(extract_string("description", character).split())
    else:
        strings.append(SpecialTokens.missing)

    return " ".join(strings[:max_length])


def summarize_cards(cards: List[Dict[str, Any]]) -> str:
    """
    Create the summary of a card
    """
    return " ".join([summarize_card(card) for card in cards])


def summarize_card(card: Optional[Dict[str, Any]]) -> str:
    """
    Create the summary of a card
    """
    strings: List[str] = [SpecialTokens.card]
    if card:
        strings.append(extract_string("name", card))
        strings.append(extract_string("description", card))
    else:
        strings.append(SpecialTokens.missing)

    return " ".join(strings)


def summarize_challenge(
    challenge: Optional[Dict[str, Any]], max_length: int = 100
) -> str:
    """
    Create the summary of a challenge

    NOTE: The default max_length parameter was chosen empirically to make the
    context size less than half the max length supported by GPT-2 (1024).
    """
    if not challenge:
        return " ".join((SpecialTokens.challenge, SpecialTokens.missing))

    return " ".join(
        chain.from_iterable(
            tuple(
                [t.value] + extract_string(f, challenge).split()[:max_length]
                for f, t in (
                    ("description", SpecialTokens.challenge),
                    ("success_stakes", SpecialTokens.success_stakes),
                    ("failure_stakes", SpecialTokens.failure_stakes),
                )
            )
        )
    )


def summarize_entry(entry: Dict[str, Any]) -> str:
    """
    Create the summary of an entry
    """
    return " ".join(
        (
            summarize_challenge(entry.get("target_challenge_card")),
            summarize_cards(entry.get("cards_played_on_challenge", [])),
        )
    )


def process_story(story: Dict[str, Any]) -> Optional[ProcessedStory]:
    """
    Summarize a scene. Returns a list of summarized scene entries.
    """
    scenes = story.get("scenes")
    characters = story.get("characters")
    if not scenes or not characters or not isinstance(scenes, Sequence):
        return None

    all_characters: Dict[str, CharacterInfo] = {}
    for character in characters:
        character_id = character.get("character_seq_id")
        if not character_id:
            # This would only happen with malformed data
            continue

        all_characters[character_id] = CharacterInfo(
            summary=summarize_character(character),
            character_id=character_id,
            entry_ids=[],
        )

    all_entries: Dict[str, EntryInfo] = {}
    for scene in scenes:
        entries = scene.get("entries", [])
        if not entries or not isinstance(entries, Sequence):
            continue

        for entry in entries:
            entry_id = entry.get("seq_id", None)
            if entry_id is None:
                # This would only happen with malformed data
                continue

            character_id = entry.get("character_seq_id")
            if not character_id:
                # Narrator moves do not have a character id, and we aren't
                # currently modeling narrator moves
                continue

            challenge = entry.get("target_challenge_card")
            cards = entry.get("cards_played_on_challenge", [])
            if not challenge or not cards:
                # Only modeling moves targeting a challenge with played cards
                continue

            all_entries[entry_id] = EntryInfo(
                text=entry.get("description", ""),
                entry_id=entry_id,
                summary=summarize_entry(entry),
                character_id=character_id,
            )

            character_info = all_characters[character_id]
            character_info.entry_ids.append(entry_id)

    return ProcessedStory(entries=all_entries, characters=all_characters)


def get_entry_summary(story: ProcessedStory, entry_id: str) -> str:
    """
    Extracts the context for a given entry
    """
    entry_info = story.entries.get(entry_id)
    if not entry_info:
        raise KeyError(f"Cannot find entry {entry_id} in story!")

    character_info = story.characters.get(entry_info.character_id)
    if not character_info:
        raise KeyError(f"Cannot find character {entry_info.character_id} in story!")

    return " ".join([character_info.summary, entry_info.summary])


def get_entry(story: ProcessedStory, entry_id: str) -> str:
    """
    Extracts the context for a given entry
    """
    entry_info = story.entries.get(entry_id)
    if not entry_info:
        raise KeyError(f"Cannot find entry {entry_id} in story!")

    return " ".join([get_entry_summary(story, entry_id), entry_info.text])


def split_dataset(data_path, splits: Tuple[int, ...]) -> Tuple[List[str], ...]:
    """
    Return a list of files that split the dataset according to these constraints:

    1) They approximately meet the passed in splits, which are treated as
    ratios by taking splits[i]/sum(splits)
    2) We balance the number of stories and token counts for each split
    according to the ratios
    """
    story_info = []
    total_files = 0.0
    total_words = 0.0
    total_scenes = 0.0
    total_entries = 0.0
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
            self.filenames.append(filename)

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
