"""
Generate samples from our Storium models
"""
import logging
from typing import Any, Dict, List, Optional, Set

import torch
from torch.nn import functional as F
from transformers import GPT2Config, PreTrainedModel

from data.parallel import StaticDataParallel
from data.preprocess import SpecialToken, get_tokenizer
from data.utils import collate, EntryList
from model import GPT2SegmentedModel


class SampleGenerator:
    """
    A class that encapsulates all the functionality needed to generate samples
    from a model
    """

    def __init__(
        self,
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the generator
        """
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.cache_dir = cache_dir
        self.tokenizer = get_tokenizer("gpt2", cache_dir=cache_dir)
        self.move_id = self.tokenizer.convert_tokens_to_ids(SpecialToken.move)
        self.separator_id = self.tokenizer.convert_tokens_to_ids(SpecialToken.separator)

        self.model: PreTrainedModel

    def load(self, checkpoint_path: str):
        """
        Load the model from the specified path
        """
        logging.info("Loading model")
        config = GPT2Config.from_pretrained(checkpoint_path)
        config.output_past = True

        model = GPT2SegmentedModel.from_pretrained(
            checkpoint_path, config=config, cache_dir=self.cache_dir
        )

        if torch.cuda.is_available():
            model = model.cuda()

        self.model = StaticDataParallel(model)

    def filter(self, logits: torch.Tensor):
        """
        Do top-k/top-p filtering of the logits

        Based on: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        top_k = min(self.top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float("inf")

        top_p = self.top_p
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float("inf")

        return logits

    def sample(
        self, entries: EntryList, length: int = 256, max_length: int = 1024
    ) -> List[str]:
        """
        Sample entry text given the list of summaries up to the specified length
        """
        with torch.no_grad():
            self.model.eval()
            batch = collate(entries)
            seq_length = batch["tokens"].shape[1]
            batch = self.sample_first(batch)

            final_length = min(seq_length + length, max_length)
            desired_length = final_length - seq_length
            done = [t.item() == self.tokenizer.eos_token_id for t in batch["tokens"]]
            outputs = [t.tolist() for t in batch["tokens"]]
            for _ in range(seq_length, final_length):
                batch = self.sample_next(batch, outputs, desired_length)
                for idx, (token, output) in enumerate(zip(batch["tokens"], outputs)):
                    next_token = token.item()
                    done[idx] = done[idx] or (next_token == self.tokenizer.eos_token_id)
                    if done[idx]:
                        continue

                    output.append(next_token)

        return [self.tokenizer.decode(output) for output in outputs]

    def sample_logits(
        self,
        logits: torch.Tensor,
        generated: Optional[List[Set[int]]] = None,
        length_penalties: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Perform sampling on the passed in logits
        """
        # First apply the repetition penalty
        if generated:
            for idx, tokens in enumerate(generated):
                for token in tokens:
                    logits[idx, token] /= self.repetition_penalty

        # Then apply a length penalty if specified
        if length_penalties:
            for idx, penalty in enumerate(length_penalties):
                logits[idx, self.tokenizer.eos_token_id] *= penalty

        # Ensure we cannot get a separator, as that shouldn't occur
        logits[:, self.separator_id] = 0

        # Then filter the logits
        temperature = self.temperature
        logits = self.filter(logits / (temperature if temperature > 0 else 1.0))

        return (
            torch.argmax(logits, dim=-1)
            if temperature == 0  # greedy sampling
            else torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)
        )

    def sample_first(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sample the first token. The first token is a special case since we pass
        in the full context and generate the "past" hidden states.
        """
        lengths = batch["lengths"]
        num_tokens = batch["num_tokens"]

        batch_size = len(lengths)
        indices = torch.LongTensor(lengths)  # type:ignore

        outputs = self.model(batch)
        next_token = self.sample_logits(outputs[0][range(batch_size), indices - 1])

        # Return an updated batch
        return {
            "past": outputs[1],
            "tokens": next_token.unsqueeze(-1),
            "segments": next_token.new_full((batch_size, 1, 1), self.move_id),
            "segment_masks": next_token.new_ones((batch_size, 1, 1)),
            "lengths": [l + 1 for l in lengths],
            "num_tokens": num_tokens + batch_size,
        }

    def sample_next(
        self, batch: Dict[str, Any], generated: List[List[int]], desired_length: int
    ) -> Dict[str, Any]:
        """
        Sample the next token for the passed in batch
        """
        lengths = batch["lengths"]
        num_tokens = batch["num_tokens"]

        batch_size = len(lengths)

        outputs = self.model(batch)
        next_token = self.sample_logits(
            outputs[0][:, 0],
            generated=[set(tokens) for tokens in generated],
            length_penalties=[len(tokens) / desired_length for tokens in generated],
        )

        # Return an updated batch
        return {
            "past": outputs[1],
            "tokens": next_token.unsqueeze(-1),
            "segments": batch["segments"],
            "segment_masks": batch["segment_masks"],
            "lengths": [l + 1 for l in lengths],
            "num_tokens": num_tokens + batch_size,
        }
