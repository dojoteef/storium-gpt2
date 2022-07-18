"""
GPT3 figmentator
"""
import logging
import os
import traceback
from typing import Any, Dict, List, Optional

import openai
from figmentator.figment.base import CharacterEntryFigmentator
from figmentator.models.figment import FigmentContext, FigmentStatus
from figmentator.models.suggestion import SuggestionType
from openai.error import RateLimitError

from data.preprocess import get_tokenizer
from data.preprocess.gpt3 import (EntryInfo, Preprocessor, ProcessedStory,
                                  SpecialToken)

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
        self.generate_args: Dict[str, Any]
        self.revise_args: Optional[Dict[str, Any]]
        self.max_entry_length: int
        self.default_retry_time: int

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
            revise_args = properties.get("revise_args", None)

            self.default_retry_time = properties.get("default_retry_time", 2)
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
        if revise_args is not None:
            revise_args["logit_bias"] = generate_args["logit_bias"]

        self.generate_args = generate_args
        self.revise_args = revise_args
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
        move = self.preprocessor.get_move(story, entry_info, add_suffix=False)
        with move.constraint(MAX_LENGTH - self.max_entry_length):
            entry = self.preprocessor.decode(move)

        logger.debug("returning %s", entry)
        return entry

    def build_edit_prompt(self, prompt: str, generated: str) -> str:
        """
        Build the prompt for the GPT-3 revision model
        """
        return prompt + generated + "\n" + str(SpecialToken.edited_move) + " "

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

        return {
            "user": entry.user_pid,
            "prompt": self.build_prompt(story, entry_info),
        }

    def should_filter(self, text: str, user: str) -> bool:
        """
        Should the text be filtered, see https://beta.openai.com/docs/engines/content-filter
        """
        if not text:
            return False

        try:
            response = openai.Completion.create(
                engine="content-filter-alpha",
                prompt=f"<|endoftext|>{text}\n--\nLabel:",
                user=user,
                temperature=0,
                max_tokens=1,
                top_p=0,
                logprobs=10,
            )
        except Exception as e:
            logger.error(str(e))
            return True

        output_label = response["choices"][0]["text"]

        # This is the probability at which we evaluate that a "2" is likely real
        # vs. should be discarded as a false positive
        toxic_threshold = -0.355

        if output_label == "2":
            # If the model returns "2", return its confidence in 2 or other output-labels
            logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]

            # If the model is not sufficiently confident in "2",
            # choose the most probable of "0" or "1"
            # Guaranteed to have a confidence for 2 since this was the selected token.
            if logprobs["2"] < toxic_threshold:
                logprob_0 = logprobs.get("0", None)
                logprob_1 = logprobs.get("1", None)

                # If both "0" and "1" have probabilities, set the output label
                # to whichever is most probable
                if logprob_0 is not None and logprob_1 is not None:
                    if logprob_0 >= logprob_1:
                        output_label = "0"
                    else:
                        output_label = "1"
                # If only one of them is found, set output label to that one
                elif logprob_0 is not None:
                    output_label = "0"
                elif logprob_1 is not None:
                    output_label = "1"

                # If neither "0" or "1" are available, stick with "2"
                # by leaving output_label unchanged.

        # if the most probable token is none of "0", "1", or "2"
        # this should be set as unsafe
        if output_label not in ["0", "1", "2"]:
            output_label = "2"

        return output_label == "2"

    def sample(self, processed: Dict[str, Any]) -> Optional[str]:
        """
        This method generates a batch of character entry text
        """
        logger.debug("Sampling continuation")
        if not processed:
            return None

        if len(processed) > 1:
            raise ValueError("This figmentator does not support batching!")

        user = processed[0]["user"]
        try:
            response = openai.Completion.create(
                user=user, prompt=processed[0]["prompt"], **self.generate_args
            )
            sample = response["choices"][0]["text"]
        except RateLimitError as rle:
            wait_time = int(rle.headers.get("Retry-After", self.default_retry_time))
            logger.warning("Hit rate limit. Waiting %d seconds", wait_time)
            sample = ""
        except openai.OpenAIError as e:
            logger.error(str(e))
            return None

        logger.debug(str(response))
        if self.should_filter(sample, user):
            logger.warning("filtering %s", sample)
            return None

        return sample

    def sample_edit(self, processed: Dict[str, Any], initial_sample: str) -> str:
        """
        This method samples an edit for the given suggestion
        """
        if self.revise_args is None:
            return initial_sample

        user = processed["user"]
        try:
            response = openai.Completion.create(
                user=user,
                prompt=self.build_edit_prompt(processed["prompt"], initial_sample),
                **self.revise_args,
            )
            revised = response["choices"][0]["text"]
            logger.debug(str(response))

            if self.should_filter(revised, user):
                logger.warning("filtering %s", revised)
                return initial_sample
        except (RateLimitError, openai.OpenAIError) as e:
            revised = initial_sample
            logger.error(str(e))

        return revised

    def figmentate(self, contexts: List[FigmentContext]) -> List[FigmentContext]:
        """
        This method should generate a figment for each context in the list.

        It returns a list of scene entries with the suggestion filled in or
        None.
        """
        if len(contexts) > 1:
            raise ValueError("This figmentator does not support batching!")

        context = contexts[0]
        segment = self.validate(context)
        if not segment:
            context.status = FigmentStatus.failed
            return contexts

        processed = self.process(context)
        if processed is None:
            context.status = FigmentStatus.failed
            return contexts

        # Make sure we filter profanity that the model might generate
        sample = self.sample(processed)
        if not sample:
            context.status = FigmentStatus.failed
            return contexts

        sample = self.profanity_filter(sample)
        if context.status == FigmentStatus.failed:
            return contexts

        assert context.range is not None
        assert context.entry.description is not None

        context.entry.description += sample
        chunks = context.range.unit.chunk(context.entry.description)

        # Mark the status as completed or partially completed
        if context.range.is_finite() and len(chunks) > segment.stop:
            context.status = FigmentStatus.completed

            if self.revise_args is not None:
                context.entry.figment_state = {"initial_sample": sample}
                context.entry.description = self.sample_edit(processed, sample)
        else:
            context.status = FigmentStatus.partial

        return contexts
