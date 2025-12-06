import dataclasses
import os
from dataclasses import dataclass
from typing import Optional

import torch as t
from jaxtyping import Float, Int
from torch import Tensor

from .config import RLHFArgs


@dataclass
class ReplayMinibatch:
    sample_ids: Int[Tensor, " minibatch_size seq_len"]
    advantages: Float[Tensor, " minibatch_size gen_len"]  # per-token advantages


class ReplayMemory:
    def __init__(
        self,
        args: RLHFArgs,
        sample_ids: Int[Tensor, " batch_size seq_len"],
        advantages: Float[Tensor, " batch_size gen_len"],
    ):
        assert sample_ids.shape[0] == args.batch_size
        assert advantages.shape == (args.batch_size, args.gen_len)
        self.args = args
        self.sample_ids = sample_ids
        self.advantages = advantages

    def get_minibatches(self) -> list[ReplayMinibatch]:
        minibatches = []
        for _ in range(self.args.batches_per_learning_phase):
            for idx in t.randperm(self.args.batch_size).reshape(self.args.num_minibatches, -1):
                minibatches.append(
                    ReplayMinibatch(
                        sample_ids=self.sample_ids[idx],
                        advantages=self.advantages[idx],
                    )
                )
        return minibatches


# Helper: concatenate memories (for multiple rollouts per phase)
def _cat_memories(memories: list["ReplayMemory"]) -> "ReplayMemory":
    assert len(memories) > 0
    args0 = memories[0].args
    sample_ids = t.cat([m.sample_ids for m in memories], dim=0)
    advantages = t.cat([m.advantages for m in memories], dim=0)

    new_args = dataclasses.replace(args0, batch_size=sample_ids.shape[0])
    new_args.minibatch_size = new_args.batch_size // new_args.num_minibatches
    return ReplayMemory(
        args=new_args,
        sample_ids=sample_ids,
        advantages=advantages,
    )


# prompts loader (path or inline)
def load_prompts(prompts_path: Optional[str], prompts_inline: Optional[list[str]]) -> list[str]:
    if prompts_inline and len(prompts_inline) > 0:
        return prompts_inline
    if prompts_path and os.path.exists(prompts_path):
        with open(prompts_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        return blocks
    return ["[no prompts given]"]

