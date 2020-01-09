"""
A very basic example of a figmentator
"""
import logging
import traceback
from typing import Any, Dict, List, Optional, Set

from figmentator.figment.base import Figmentator
from figmentator.models.figment import FigmentContext
from figmentator.models.storium import SceneEntry
from figmentator.models.suggestion import SuggestionType

from data.preprocess import Preprocessor, tensorize
from sample import SampleGenerator


class GPT2Figmentator(Figmentator):
    """
    Figmentator for GPT2 baseline models generates scene_entry suggestions
    """

    def __init__(self, suggestion_type: SuggestionType):
        """ Initialize the figmentator """
        if suggestion_type is not SuggestionType.scene_entry:
            raise ValueError("This figmentator can only generate scene entries!")

        super().__init__(suggestion_type)

        self.generator: SampleGenerator
        self.preprocessor: Preprocessor

    def startup(self, properties: Optional[Dict[str, Any]] = None) -> bool:
        """
        This method should perform any necessary startup, such as loading the model
        parameters. After this method completes, the model should be ready to perform
        preprocessing and suggestion generation. It should return whether it was able to
        successfully startup.
        """
        if not properties:
            return False

        try:
            # First create and load the generator
            generator = SampleGenerator(**properties.get("sample", {}))
            generator.load(properties["checkpoint_path"])

            # Then the preprocessor
            preprocessor = Preprocessor(
                generator.tokenizer, **properties.get("preprocess", {})
            )
        except Exception:  # pylint:disable=broad-except
            logging.error(traceback.format_exc())
            return False

        # Finally set the class attributes
        self.generator = generator
        self.preprocessor = preprocessor

        return True

    def shutdown(self) -> None:
        """
        This method should perform any necessary shutdown actions, such as releasing the
        model parameters. After this method completes, all resources used by the model
        should be released.
        """
        if hasattr(self, "generator"):
            del self.generator

        if hasattr(self, "preprocessor"):
            del self.preprocessor

    def preprocess(
        self, story_snapshot: Dict[str, Any], data: Optional[Any] = None
    ) -> Any:
        """
        This method should perform any preprocessing required on the story needed before
        generating suggestions. It should return an object representing the
        preprocessed story. This object will be provided to the figmentate method.

        - story: A story as specified in https://storium.com/help/export/json/0.9.2
        - data: an optional object representing any previously preprocesed data from a
          previous snapshot of the same story
        """
        return self.preprocessor.process_story(story_snapshot, processed=data)

    def figmentate(self, contexts: List[FigmentContext]) -> List[Optional[SceneEntry]]:
        """
        This method should generate a figment for each context in the list.
        """
        lengths: List[int] = []
        entries: List[SceneEntry] = []
        generated: List[Set[int]] = []
        processed_entries: List[Dict[str, Any]] = []
        for context in contexts:
            story = context.data
            entry = context.entry.copy()
            entries.append(None)

            if not context.range:
                logging.warning("Failed to generate text: no range specified")
                continue

            if len(context.range.ranges) > 1:
                logging.warning("Failed to generate text: too many ranges specified")
                continue

            text_range = context.range.ranges[0]
            if not text_range.end:
                logging.warning("Failed to generate text: no range end specified")
                continue

            num_tokens = (
                text_range.end - text_range.start
                if text_range.start
                else text_range.end
            )
            max_length = self.preprocessor.max_length - num_tokens

            entry_info = self.preprocessor.process_entry(
                entry.dict(),
                story.establishment_entries.indices[-1],
                0,
                add_eos=False,
                force=True,
            )
            if not entry_info:
                logging.warning("Unable to process entry: Failed to generate text.")
                continue

            entries[-1] = entry
            lengths.append(num_tokens)
            generated.append(
                set(self.preprocessor.tokenizer.encode(entry.description or ""))
            )
            move = self.preprocessor.get_move(story, entry_info)
            with move.constraint(max_length, naive=self.preprocessor.naive_layout):
                processed_entries.append(tensorize(move.asdict()))  # type:ignore

        samples = self.generator.sample(processed_entries, generated, lengths)
        for entry, sample in zip(entries, samples):
            entry.description = (entry.description or "") + sample

        return entries
