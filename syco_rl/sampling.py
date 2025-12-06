from typing import Optional

import torch as t
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer

from eindex import eindex


# def reward_fn_char_count(generated_samples: list[str], char: str = ".") -> t.Tensor:
#     """Unused now (kept for debugging): count occurrences of `char` in each string."""
#     return t.tensor([s.count(char) for s in generated_samples], device=device, dtype=t.float32)


def get_samples(
    model: HookedTransformer,
    prompts: list[str],
    gen_len: int = 15,
    temperature: float = 0.8,
    top_k: Optional[int] = 15,
    prepend_bos: bool = True,
    **kwargs,
) -> tuple[Int[Tensor, "batch seq"], list[str]]:
    input_ids = model.to_tokens(prompts, prepend_bos=prepend_bos)  # [B, S]
    gen_kwargs = dict(max_new_tokens=gen_len, stop_at_eos=False, temperature=temperature)
    if top_k is not None:
        gen_kwargs["top_k"] = top_k
    output_ids = model.generate(input_ids, **gen_kwargs)
    samples = model.to_string(output_ids)
    return output_ids.clone(), samples


def get_logprobs(
    logits: Float[Tensor, "batch seq_len vocab"],
    tokens: Int[Tensor, "batch seq_len"],
    prefix_len: int | None = None,
) -> Float[Tensor, "batch gen_len"]:
    """
    Calculates the logprobs for the *generated* tokens.
    - logits[b, i] is the distribution for tokens[b, i+1]
    
    [FIXED (Comment 8)]
    """
    
    if prefix_len is not None:
        # We want to score the generated tokens, which start at index `prefix_len`.
        # The logits for the *first* generated token (at index `prefix_len`)
        # come from the *last* prompt token (at index `prefix_len - 1`).
        
        # Logits from the end of the prompt to the second-to-last token
        # Shape: [B, gen_len, V]
        logits_to_use = logits[:, prefix_len - 1 : -1]
        
        # Generated tokens (the ones we want to score)
        # Shape: [B, gen_len]
        tokens_to_predict = tokens[:, prefix_len:]
    
    else:
        # No prefix, score all tokens except the first one
        logits_to_use = logits[:, :-1]
        tokens_to_predict = tokens[:, 1:]

    assert logits_to_use.shape[1] == tokens_to_predict.shape[1], \
        f"Logits and tokens shape mismatch: {logits_to_use.shape[1]} vs {tokens_to_predict.shape[1]}"

    logprobs = logits_to_use.log_softmax(-1)
    
    # We want logprobs[b, s, tokens_to_predict[b, s]]
    correct_logprobs = eindex(logprobs, tokens_to_predict, "b s [b s]")
    
    return correct_logprobs

