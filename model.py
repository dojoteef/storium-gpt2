"""
Define the baseline model we'll use
"""
from typing import Any, Dict

import torch
from transformers import GPT2LMHeadModel


class GPT2SegmentedModel(GPT2LMHeadModel):
    """
    Our baseline model which uses composable segments
    """

    @property
    def wte(self):
        """
        Get the weights for the token embeddings from the transformer
        """
        return self.transformer.wte

    def forward(self, inputs: Dict[str, Any]):  # pylint:disable=arguments-differ
        """
        Compose the segments together and call the base class
        """
        tokens = inputs["tokens"]
        segment_masks = inputs["segment_masks"].unsqueeze(-1)
        segments_embeds = self.wte(inputs["segments"]) * segment_masks
        inputs_embeds = self.wte(
            torch.max(tokens, tokens.new_zeros(1))
        ) + segments_embeds.sum(1)

        return super().forward(inputs_embeds=inputs_embeds, labels=tokens)
