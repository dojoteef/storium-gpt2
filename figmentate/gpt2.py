"""
GPT2 figmentator
"""
import logging
import traceback
from typing import Any, Dict, List, Optional

from figmentator.figment.base import CharacterEntryFigmentator
from figmentator.models.figment import FigmentContext
from figmentator.models.suggestion import SuggestionType

from data.preprocess import tensorize
from data.preprocess.gpt2 import Preprocessor
from sample import SampleGenerator


class GPT2Figmentator(CharacterEntryFigmentator):
    """
    Figmentator for GPT2 baseline models that generate scene_entry suggestions
    """

    def __init__(self, suggestion_type: SuggestionType):
        """Initialize the figmentator"""
        if suggestion_type is not SuggestionType.scene_entry:
            raise ValueError("This figmentator can only generate scene entries!")

        super().__init__(suggestion_type)

        self.rate: int
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

            # Get the number of tokens to create per update
            rate = properties.get("rate", 25)
        except Exception:  # pylint:disable=broad-except
            logging.error(traceback.format_exc())
            return False

        # Finally set the class attributes
        self.rate = rate
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

    def process(self, context: FigmentContext) -> Optional[Dict[str, Any]]:
        """
        This method performs any processing needed before generating a suggestion
        """
        entry = context.entry
        story = context.data
        entry_info = self.preprocessor.process_entry(
            entry.dict(),
            story.establishment_entries.indices[-1],
            0,
            add_eos=False,
            force=True,
        )
        if not entry_info:
            logging.warning("Unable to process entry: Failed to generate text.")
            return None

        move = self.preprocessor.get_move(story, entry_info)
        with move.constraint(
            self.preprocessor.max_length - self.rate,
            naive=self.preprocessor.naive_layout,
        ):
            return {
                "tokens": tensorize(move.asdict()),
                "generated": set(
                    self.preprocessor.tokenizer.encode(entry.description or "")
                ),
                "length": self.rate,
            }

    def sample(self, processed: List[Dict[str, Any]]) -> List[str]:
        """
        This method generates a batch of character entry text
        """
        if not processed:
            return []

        return self.generator.sample(*zip(*(p.values() for p in processed)))
