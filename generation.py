# https://github.com/huggingface/transformers/blob/698c9e2dbdbc35aed588b58f080afbdbfa0c3c04/src/transformers/generation_utils.py
# https://github.com/huggingface/transformers/blob/698c9e2dbdbc35aed588b58f080afbdbfa0c3c04/src/transformers/generation_logits_process.py
# simpler version of GenerationMixin from huggingface for good inference

import torch
from torch.nn import functional as F

from transformers.generation_logits_process import (
    TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, LogitsProcessorList
)
from transformers.generation_beam_search import (
 BeamSearchScorer, BeamHypotheses
)

class BeamSearchScorer():
  def __init__(
    self,
    batch_size: int,
    max_length: int,
    num_beams: int,
    device: torch.device,
    length_penalty: float = 1.0,
    do_early_stopping: bool = False,
    num_beam_hyps_to_keep: int = 1,
    num_beam_groups: int = 1,
  ):
    self.max_length = max_length
    self.num_beams = num_beams
    self.device = device
    self.length_penalty = length_penalty
    self.do_early_stopping = do_early_stopping
    self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
    self.num_beam_groups = num_beam_groups
    self.group_size = self.num_beams // self.num_beam_groups

    self._is_init = False
    self._beam_hyps = [
      BeamHypotheses(
        num_beams=self.num_beams,
        max_length=self.max_length,
        length_penalty=self.length_penalty,
        early_stopping=self.do_early_stopping,
      )
      for _ in range(batch_size)
    ]

    # since we are generating upto a fixed size this may not really be needed but still added
    # self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

  def process(
    self,
    text_tokens: torch.LongTensor,
    next_scores: torch.FloatTensor,
    next_tokens: torch.LongTensor,
    next_indices: torch.LongTensor,
  ):
    batch_size = len(self._beam_hyps)
    assert batch_size == (text_tokens.size(0) // self.group_size)

    device = text_tokens.device
    next_beam_scores = torch.zeros([batch_size, self.group_size], dtype = next_scores.dtype, device = device)
    next_beam_tokens = torch.zeros([batch_size, self.group_size], dtype = next_tokens.dtype, device = device)
    next_beam_indices = torch.zeros([batch_size, self.group_size], dtype = next_indices.dtype, device = device)

    for batch_idx, beam_hyp in enumerate(self._beam_hyps):
      # skipping done check because full generation required

      # next tokens for this sentence
      beam_idx = 0
      for beam_token_rank, (next_token, next_score, next_index) in enumerate(
        zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
      ):
        batch_beam_idx = batch_idx * self.group_size + next_index

        # ~~add to generated hypothesis if end of sentence~~ not needed again

        # add next predicted token since it is not eos_token
        next_beam_scores[batch_idx, beam_idx] = next_score
        next_beam_tokens[batch_idx, beam_idx] = next_token
        next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
        beam_idx += 1

        # once the beam for next step is full, don't add more tokens to it.
        if beam_idx == self.group_size:
            break

    return {
      "next_beam_scores": next_beam_scores.view(-1),
      "next_beam_tokens": next_beam_tokens.view(-1),
      "next_beam_indices": next_beam_indices.view(-1),
    }

  def finalize(
    self,
    input_ids: torch.LongTensor,
    final_beam_scores: torch.FloatTensor
  ):
    batch_size = len(self._beam_hyps)
    device = input_ids.device

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_idx, beam_hyp in enumerate(self._beam_hyps):

      # all open beam hypotheses are added to the beam hypothesis
      # beam hypothesis class automatically keeps the best beams
      for beam_id in range(self.num_beams):
        batch_beam_idx = batch_idx * self.num_beams + beam_id
        final_score = final_beam_scores[batch_beam_idx].item()
        final_tokens = input_ids[batch_beam_idx]
        beam_hyp.add(final_tokens, final_score)

    # select the best hypotheses
    sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
    best = []
    best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=device, dtype=torch.float32)

    # retrieve best hypotheses
    for i, beam_hyp in enumerate(self._beam_hyps):
      sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
      for j in range(self.num_beam_hyps_to_keep):
        best_hyp_tuple = sorted_hyps.pop()
        best_score = best_hyp_tuple[0]
        best_hyp = best_hyp_tuple[1]
        sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

        # append to lists
        best.append(best_hyp)
        best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

    return {
      "sequences": torch.cat(best, dim = 0).view(len(best), -1),
      "sequence_scores": best_scores
    }


class GenerationMixin():
  def __init__(self, config):
    self.config = config

  @staticmethod
  def _expand_tokens_for_generation(
    input_ids,
    expand_size,
  ):
    expanded_return_idx = (
      torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
    )
    input_ids = input_ids.index_select(0, expanded_return_idx)
    return input_ids


  @staticmethod
  def _get_logits_warper_list(num_beams, temperature, top_k, top_p):
    warpers = LogitsProcessorList()
    if temperature is not None and temperature != 1.0:
      warpers.append(TemperatureLogitsWarper(temperature))

    if top_k is not None and top_k != 0:
      warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1)))

    if top_p is not None and top_p < 1.0:
      warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
    return warpers

  def beam_sample(
    self,
    text_tokens,
    image_tokens,
    beam_scorer,
    logits_warper,
    steps_to_gen,
    batch_size,
    num_beams,
    output_attentions=False,
    output_hidden_states=False,
    _verbose = False
  ):
    batch_beam_size, cur_len = text_tokens.shape
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=text_tokens.device)
    beam_scores = beam_scores.view((batch_size * num_beams,))

    for s in range(steps_to_gen):
      # for the number of steps to generate
      model_inputs = {
        "text_tokens": text_tokens,
        "image_tokens": image_tokens
      }

      outputs = self(
        **model_inputs,
        return_dict=True,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
      )

      next_token_logits = outputs[0][:, -1, :] # logits is always the first
      next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
      next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

      # all three LogitsWarpers we are using has no use of first argument ie. input_ids, so we are passing None
      next_token_scores = logits_warper(None, next_token_scores)

      # reshape for beam search
      vocab_size = next_token_scores.shape[-1]
      next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

      probs = F.softmax(next_token_scores, dim = -1) # [bs, nb * vocab]
      try:
        next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
      except:
        print(probs)
      next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

      next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim =1) 
      next_tokens = torch.gather(next_tokens, -1, _indices)

      next_indices = next_tokens // vocab_size
      next_tokens = next_tokens % vocab_size

      # stateless
      # https://github.com/huggingface/transformers/blob/698c9e2dbdbc35aed588b58f080afbdbfa0c3c04/src/transformers/generation_beam_search.py#L199
      beam_outputs = beam_scorer.process(
        text_tokens = text_tokens,
        next_scores=next_token_scores,
        next_tokens=next_tokens,
        next_indices=next_indices,
      )

      beam_scores = beam_outputs["next_beam_scores"]
      beam_next_tokens = beam_outputs["next_beam_tokens"]
      beam_idx = beam_outputs["next_beam_indices"]

      if _verbose:
        print("image_tokensimage_tokensimage_tokensimage_tokensimage_tokens", image_tokens)
        print("beam_next_tokensbeam_next_tokensbeam_next_tokensbeam_next_tokens", beam_next_tokens)
        print("beam_idxbeam_idxbeam_idx", beam_idx)

      beam_tokens_to_append = beam_next_tokens.view(-1, 1)
      if image_tokens is not None:
        image_tokens = torch.cat([image_tokens[beam_idx, :], beam_tokens_to_append], dim=-1)
      else:
        image_tokens = beam_tokens_to_append
      model_inputs["image_tokens"] = image_tokens

    sequence = beam_scorer.finalize(
      input_ids = image_tokens,
      final_beam_scores=beam_scores,
    )

    return sequence

  @torch.no_grad()
  def complete_image(
    self,
    text_tokens,
    image_tokens = None,
    num_return_sequences=1,
    num_beams=1,
    top_k = None,
    top_p = None,
    temperature = None,
    _verbose = False
  ):
    config = self.config
    processors = None # there are no pre-processors that we need
    batch_size = text_tokens.shape[0] * num_return_sequences
    device = text_tokens.device

    # we have an equivalent of is_beam_sample_gen_mode
    logits_warper_list = self._get_logits_warper_list(num_beams, temperature, top_k, top_p)

    # this model always generates to a fixed size
    steps_to_gen = config.total_context_len - config.text_context_len
    if image_tokens is not None:
      steps_to_gen -= image_tokens.shape[1]
    
    if _verbose:
      print("steps_to_gen:", steps_to_gen)
      print(steps_to_gen, image_tokens, config.total_context_len)

    beam_scorer = BeamSearchScorer(
      batch_size=batch_size,
      max_length=steps_to_gen,
      num_beams=num_beams,
      device=device,
      length_penalty=1.0,
      do_early_stopping=False,
    )

    # expand text_tokens and image_tokens to num_beams

    expanded_text_tokens = self._expand_tokens_for_generation(
      text_tokens,
      expand_size=num_beams * num_return_sequences
    )
    if image_tokens is not None:
      expanded_image_tokens = self._expand_tokens_for_generation(
          image_tokens,
          expand_size=num_beams * num_return_sequences
      )
    else:
      expanded_image_tokens = None

    image_tokens = self.beam_sample(
      text_tokens = expanded_text_tokens,
      image_tokens = expanded_image_tokens,
      beam_scorer = beam_scorer,
      logits_warper= logits_warper_list,
      steps_to_gen=steps_to_gen,
      batch_size= batch_size,
      num_beams = num_beams,
      _verbose=_verbose
    )
    if _verbose: print("final image tokens", image_tokens["sequences"], image_tokens["sequences"][0].size())
    recons = self.vae._decode_ids(image_tokens=image_tokens["sequences"]).permute((0, 2, 3, 1))
    return recons, image_tokens["sequence_scores"]

