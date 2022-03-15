"""
GPT3 figmentator
"""
import logging
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import openai
from figmentator.figment.base import CharacterEntryFigmentator
from figmentator.models.figment import FigmentContext
from figmentator.models.suggestion import SuggestionType
from kiwisolver import UnsatisfiableConstraint

from data.preprocess import get_tokenizer
from data.preprocess.gpt3 import EntryInfo, Preprocessor, ProcessedStory

# GPT-3 can process at most 2048 tokens
MAX_LENGTH = 2048
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOGLEVEL", "info").upper())


class GPT3Figmentator(CharacterEntryFigmentator):
    """
    Figmentator for GPT3 models that generate scene_entry suggestions
    """

    def __init__(self, suggestion_type: SuggestionType):
        """Initialize the figmentator"""
        if suggestion_type is not SuggestionType.scene_entry:
            raise ValueError("This figmentator can only generate scene entries!")

        super().__init__(suggestion_type)

        self.preprocessor: Preprocessor
        self.logit_bias: Dict[str, int]

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
            # Load the tokenizer and preprocessor
            preprocessor = Preprocessor(
                get_tokenizer("gpt2"), **properties.get("preprocess", {})
            )

            # Load the arguments for the figmentator
            disallowed = properties.get("disallowed", [])
            generate_args = properties.get("generate_args", {})
        except Exception:  # pylint:disable=broad-except
            logger.error(traceback.format_exc())
            return False

        # Finally set the class attributes
        self.preprocessor = preprocessor

        # max_tokens defaults to 16 in the GPT-3 API
        self.max_entry_length = generate_args.get("max_tokens", 16)

        # Set the logit bias for disallowed tokens:
        generate_args["logit_bias"] = {
            str(t): -100 for s in disallowed for t in self.preprocessor.encode_token(s)
        }
        self.generate_args = generate_args
        logger.info("Successfully completed startup!")

        return True

    def shutdown(self) -> None:
        """
        This method should perform any necessary shutdown actions, such as releasing the
        model parameters. After this method completes, all resources used by the model
        should be released.
        """
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

    def build_prompt(self, story: ProcessedStory, entry_info: EntryInfo) -> str:
        """
        Build the prompt to send to GPT-3
        """
        move = self.preprocessor.get_move(story, entry_info)
        with move.constraint(MAX_LENGTH - self.max_entry_length):
            entry = self.preprocessor.decode(move)

        logger.debug("returning %s", entry)
        return entry

    def process(self, context: FigmentContext) -> Optional[Dict[str, Any]]:
        """
        This method performs any processing needed before generating a suggestion
        """
        entry = context.entry
        story = context.data
        logger.debug("entry info:\n%s", entry.json())
        try:
            entry_info = self.preprocessor.process_entry(
                entry.dict(),
                story.establishment_entries.indices[-1],
                0,
                force=True,
            )
        except Exception as e:
            logger.error("Unable to process entry")
            raise e

        if not entry_info:
            logger.error("Unable to process entry: Failed to generate text.")
            return None

        return {"prompt": self.build_prompt(story, entry_info)}

    def sample(self, processed: List[Dict[str, Any]]) -> List[str]:
        """
        This method generates a batch of character entry text
        """
        logger.debug("Sampling continuation")
        if not processed:
            return []

        if len(processed) > 1:
            raise ValueError("This figmentator does not support batching!")

        try:
            response = openai.Completion.create(
                prompt=processed[0]["prompt"], **self.generate_args
            )
        except Exception as e:
            logger.error(str(e))
            raise e

        logger.debug(str(response))
        return [response["choices"][0]["text"]]
