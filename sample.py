"""
Generate samples from our Storium models
"""
import logging
from typing import Any, Dict, List, Optional, Set, Union

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
        top_k: int = 0,
        top_p: float = 0.9,
        temperature: float = 0.7,
        repetition_penalty: float = 1.0,
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
        self,
        entries: EntryList,
        generated: Optional[List[Set[int]]] = None,
        lengths: Union[int, List[int]] = 256,
        max_length: int = 1024,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Sample entry text given the list of summaries up to the specified length
        """
        if not generated:
            generated = [set() for _ in range(len(entries))]

        if isinstance(lengths, int):
            lengths = [lengths] * len(entries)

        entry_lengths = [len(e["tokens"]) for e in entries]
        desired_lengths = [
            min(s + l, max_length) for s, l in zip(entry_lengths, lengths)
        ]
        num_steps = max(l - s for s, l in zip(entry_lengths, desired_lengths))

        with torch.no_grad():
            self.model.eval()
            batch = self.sample_first(collate(entries), generated)
            outputs = [t.tolist() for t in batch["tokens"]]
            generated = [g | set(o) for g, o in zip(generated, outputs)]
            done = [t.item() == self.tokenizer.eos_token_id for t in batch["tokens"]]

            # Already completed one step of sampling above, so decrement steps by 1
            for _ in range(num_steps - 1):
                batch = self.sample_next(batch, generated, desired_lengths)
                for idx, (token, output) in enumerate(zip(batch["tokens"], outputs)):
                    next_token = token.item()
                    done[idx] = done[idx] or (next_token == self.tokenizer.eos_token_id)
                    if done[idx]:
                        continue

                    output.append(next_token)
                    generated[idx].add(next_token)

        return [
            self.tokenizer.decode(output, skip_special_tokens=skip_special_tokens)
            for output in outputs
        ]

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

    def sample_first(
        self, batch: Dict[str, Any], generated: List[Set[int]]
    ) -> Dict[str, Any]:
        """
        Sample the first token. The first token is a special case since we pass
        in the full context and generate the "past" hidden states.
        """
        lengths = batch["lengths"]
        indices = torch.LongTensor(lengths)  # type:ignore

        batch_size = len(lengths)
        outputs = self.model(batch)
        next_token = self.sample_logits(
            outputs[0][range(batch_size), indices - 1], generated=generated
        )

        # Return an updated batch
        return {
            "past": outputs[1],
            "tokens": next_token.unsqueeze(-1),
            "segments": next_token.new_full((batch_size, 1, 1), self.move_id),
            "segment_masks": next_token.new_ones((batch_size, 1, 1)),
            "lengths": [l + 1 for l in lengths],
            "num_tokens": batch["num_tokens"] + batch_size,
        }

    def sample_next(
        self,
        batch: Dict[str, Any],
        generated: List[Set[int]],
        desired_lengths: List[int],
    ) -> Dict[str, Any]:
        """
        Sample the next token for the passed in batch
        """
        outputs = self.model(batch)
        next_token = self.sample_logits(
            outputs[0][:, 0],
            generated=generated,
            length_penalties=[
                len(tokens) / l for tokens, l in zip(generated, desired_lengths)
            ],
        )

        # Return an updated batch
        lengths = batch["lengths"]
        return {
            "past": outputs[1],
            "tokens": next_token.unsqueeze(-1),
            "segments": batch["segments"],
            "segment_masks": batch["segment_masks"],
            "lengths": [l + 1 for l in lengths],
            "num_tokens": batch["num_tokens"] + len(lengths),
        }
