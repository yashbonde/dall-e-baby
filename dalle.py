# this file has the transformer model

import os
import math
import json
import torch
import numpy as np
from torch import nn
from PIL import Image
from tqdm import trange
from textwrap import wrap
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from types import SimpleNamespace
from argparse import ArgumentParser
from torch.nn import functional as F
from torch.utils.data import DataLoader
from discrete_vae import VQVAE_v3, transforms, set_seed

from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import Block, MLP


WANDB = os.getenv("WANDB")
if WANDB:
  import wandb

plt.rcParams.update({
  'font.family': 'barlow',
  'font.size': 10
})

# ------ helper functions

def configure_optimizers(model, lr, weight_decay = 0.1, betas=(0.9, 0.999)):
    """
    from karpathy/minGPT
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, MLP, Block)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding) # add denorm here
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if "vae" in fpn:
              continue

            pn_type = pn.split(".")[-1]
            if pn_type == 'bias':
                # all biases will not be decayed
                no_decay.add(fpn)
            elif (
                (pn_type == 'weight' and isinstance(m, blacklist_weight_modules)) or
                "ln" in fpn or
                "positional_encoding" in fpn
            ):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn_type == 'weight' and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters() if "vae" not in pn}
    inter_params = decay & no_decay
    union_params = decay | no_decay

    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
    return optimizer


def init_weights(dalle, v=False):
  # function to initiliase the parameters, .apply() has some problems like
  # you cannot find the exact name and so there might be a "Linear" in part
  # of network you do not want to modify. This looped approach gives a much
  # better control over what happens
  for n, p in dalle.named_parameters():
    if "vae" in n:
      # we do not want to reinit VAE
      continue

    m0 = f"{p.mean().item():.3f}, {p.std().item():.3f}"
    if "weight" in n and "ln_" not in n:
      # this is linear weight
      p.data.normal_(mean=0.0, std=0.2)

    # layer norm
    elif "ln_" in n:
      if "bias" in n:
        p.data.zero_()
      else:
        p.data.fill_(1.0)

    elif "positional_encoding" in n:
      # this is the positional encoding that was giving nightmares
      # if you do not initialise it the network passes the `torch.empty()`
      # through the network without giving a warning and keeps returning
      # nan.
      p.data.normal_(mean=0.0, std=0.2)

    elif "bias" in n:
      p.data.zero_()

    m1 = f"{p.mean().item():.3f}, {p.std().item():.3f}"
    if v: print(n, "\t", m0, "\t", m1)

# ------ model classes

class Vqvae:
  """
  Wrapper for model in discrete_vae, automatically infers the architecture from model path
  """
  def __init__(self, model_path):
    # model_path looks like this path/to/vqvae3_128_325_3025_0_64
    args = self.infer_details_from_name(model_path)
    for k,v in vars(args).items():
      setattr(self, k, v)

  @staticmethod
  def infer_details_from_name(model_path):
    attrs=model_path.split("/")[-1].split("_")
    in_channels=3                    # number of channels in the image (def=3)
    input_res=int(attrs[1])          # input resolution of the image
    embedding_dim=int(attrs[2])*3    # embedding dimension for the latent space
    num_embeddings=int(attrs[3])     # number of embeddings in the codebook
    add_residual=bool(int(attrs[4])) # to use the model with residual connection or not
    dim = int(attrs[2])
    hidden_dims=[dim, int(1.5 * dim), dim * 2] # hidden dimensions for different layers
    args = SimpleNamespace(
      model_path=model_path,
      in_channels=in_channels,
      input_res=input_res,
      embedding_dim=embedding_dim,
      num_embeddings=num_embeddings,
      add_residual=add_residual,
      hidden_dims=hidden_dims,
    )
    return args

  def get_model(self):
    model = VQVAE_v3(
        in_channels=self.in_channels,
        embedding_dim=self.embedding_dim,
        num_embeddings=self.num_embeddings,
        hidden_dims=self.hidden_dims,
        add_residual=self.add_residual,
    )
    map_location = "cpu" if not torch.cuda.is_available() else "cuda"
    model.load_state_dict(torch.load(self.model_path, map_location=map_location))
    model.eval()
    return model


class TransformerConfig():
  def __init__(
    self,
    text_context_len=256,
    image_context_len=256,
    text_vocab_size=16000,
    image_vocab_size=3025,
    n_embd=512,
    n_layers=12,
    n_heads=64,
  ):
    self.text_context_len=text_context_len
    self.image_context_len=image_context_len
    self.text_vocab_size=text_vocab_size
    self.image_vocab_size=image_vocab_size
    self.n_embd=n_embd
    self.n_layers=n_layers
    self.n_heads=n_heads
    self.total_context_len = text_context_len + image_context_len


class Transformer(nn.Module):
  def __init__(self, total_context_len:int,  n_embd: int, n_layers: int, n_heads: int, attn_mask: torch.Tensor = None):
    super().__init__()
    self.n_embd = n_embd
    self.n_layers = n_layers
    self.transconfig = GPT2Config(
      n_positions=total_context_len,
      n_ctx=total_context_len,
      n_embd=n_embd,
      n_head=n_heads,
      n_layer=n_layers
    )
    self.attn_mask = attn_mask

    # using the ResidualAttentionBlock from OpenAI was causing problems
    # so using the huggingface GPT2 Block instead
    self.h = nn.ModuleList([
      Block(self.transconfig.n_ctx, self.transconfig, scale=True)
      for _ in range(n_layers)
    ])

  def forward(self, x: torch.Tensor, attn_mask = None, output_attentions = False):
    hidden_states = x
    for blk in self.h:
      output = blk(
        hidden_states,
        attention_mask=self.attn_mask if attn_mask is None else attn_mask,
        output_attentions=output_attentions,
      )
      hidden_states = output[-1]
    return [hidden_states]


class DallETransformer(nn.Module):
  def __init__(self, vae, transformer_config):
    super().__init__()
    self.vae = vae
    self.vae.requires_grad_ = False

    # transformer
    tconf = transformer_config
    self.config = tconf
    self.context_length=tconf.text_context_len + tconf.image_context_len
    total_vocab_size = tconf.text_vocab_size + tconf.image_vocab_size

    # this does not need to be a nn.Embedding because the full length will always be used
    self.positional_encoding = nn.Parameter(torch.empty(self.context_length, tconf.n_embd))
    self.token_embedding = nn.Embedding(total_vocab_size, tconf.n_embd)
    self.transformer = Transformer(
      total_context_len=self.context_length,
      n_embd=tconf.n_embd,
      n_layers=tconf.n_layers,
      n_heads=tconf.n_heads,
      attn_mask=None
    )
    self.image_head = nn.Sequential(
      nn.LayerNorm(tconf.n_embd),
      nn.Linear(tconf.n_embd, tconf.image_vocab_size)
    )

  def build_attention_mask(self):
      # lazily create causal attention mask, with full attention between the vision tokens
      # pytorch uses additive attention mask; fill with -inf
      mask = torch.ones(self.context_length, self.context_length) * -1e6
      # mask.fill_(-1e6) # nan happens with float("-inf")
      mask.triu_(1)      # zero out the lower diagonal
      return mask.unsqueeze(0).unsqueeze(0)

  def forward(self, text_tokens, images = None, image_tokens=None, attn_mask = None, recons = False, loss = False):
    """this model automanages the text tokens by incrementing to the correct vocab"""
    config = self.config
    no_image = True
    if image_tokens is None and images is not None:
      with torch.no_grad():
        image_tokens = self.vae._encode_image(images) # [B,i]
        no_image = False
    elif image_tokens is not None:
      no_image = False

    # increment because the image tokens occupy the first segement
    text_tokens = text_tokens + config.image_vocab_size

    if no_image:
      tokens = text_tokens
    else:
      tokens = torch.cat([text_tokens, image_tokens], dim = -1) # [B,t] + [B,i] = [B,M]

    total_gen = tokens.shape[1]
    embed = self.token_embedding(tokens) + self.positional_encoding[:total_gen, :] # [B,M,e]

    if attn_mask is not None:
      attn_mask = attn_mask.view(embed.size(0), -1)
      attn_mask = attn_mask[:, None, None, :]
      attn_mask = (1.0 - attn_mask) * -10000.0
    else:
      attn_mask = self.build_attention_mask().to(embed.device)

    # transformer blocks return tuple [hidden_states, present, (attentions, cross_attentions)]
    out = self.transformer(x = embed, attn_mask = attn_mask)[0] # [B,M,e]
    out = self.image_head(out) # [B,M,vi]
    output = [out]

    if loss:
      # note that we do not need to calculate loss for the entire sequence but only for the image tokens
      # so the labels are -100 for text_tokens and image_tokens concatenated
      labels = torch.cat([
        torch.ones_like(text_tokens).long() * -100,
        image_tokens
      ], dim = -1)[:, 1:].contiguous().view(-1)
      logits = out[:, :-1].contiguous().view(-1, config.image_vocab_size)
      loss = F.cross_entropy(logits, labels, ignore_index=-100)
      output = [out, loss]

    if recons:
      # caller wants to see the constructed image
      softmax = out[:, len(text_tokens[0]):].softmax(dim = -1) # [B, HW, E]
      image_dim = np.sqrt(softmax.shape[1]).astype(int)
      softmax = softmax.view(-1, image_dim, image_dim, softmax.size(-1))
      softmax = softmax.permute((0, 3, 1, 2)) # [B,H,W,E] -> [B,E,H,W]
      image_gen_tokens = F.one_hot(
        torch.argmax(softmax, dim = 1),
        num_classes = softmax.size(1)
      ).permute((0, 3, 1, 2)).float()
      recons = self.vae._decode_ids(image_gen_tokens)
      output = output + [recons]

    return output


class Dalle():
  def __init__(self, model_args, vae, tokenizer):
    self.model = DallETransformer(vae, model_args)
    self.device = "cpu"
    self.model.load_state_dict(torch.load(
      model_args.model_path,
      map_location=self.device
    ))
    self.model.eval()

    self.model_args = model_args
    self.tokenizer = tokenizer
    self.text_end_id = self.tokenizer.get_vocab()["<|endoftext|>"]
    self.image_end_id = self.tokenizer.get_vocab()["<|endofimage|>"]

  
  @staticmethod
  def parse_name(model_path, text_context_len, image_vocab_size, text_vocab_size):
    # folder_path = f"./dalle_{vqvae_arch.input_res}_{args.n_embd}_"
    #   "{args.n_layers}_{args.n_heads}_{args.batch_size}/dalle_{gs}.pt"
    gs = int(model_path.split("/")[-1].split("_")[-1].split(".")[0])
    arch = model_path.split("/")[-2]
    res = int(arch.split("_")[1])
    return SimpleNamespace(
      model_path=model_path,
      _gs = gs,
      input_res=res,
      n_embd=int(arch.split("_")[2]),
      n_layers=int(arch.split("_")[3]),
      n_heads=int(arch.split("_")[4]),
      batch_size=int(arch.split("_")[5]),
      text_context_len=text_context_len,
      image_vocab_size=image_vocab_size,
      image_context_len=int((res / 8) ** 2),
      text_vocab_size=text_vocab_size,
      total_context_len= text_context_len + int((res / 8) ** 2)
    )

  # most of the code for generation comes from hugginface
  # https://github.com/huggingface/transformers/src/transformers/generation_utils.py
  @staticmethod
  def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
  ):
    if top_k > 0:
      top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
      # Remove all tokens with a probability less than the last token of the top-k
      indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
      logits[indices_to_remove] = filter_value

    if top_p < 1.0:
      sorted_logits, sorted_indices = torch.sort(logits, descending=True)
      cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

      # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
      sorted_indices_to_remove = cumulative_probs > top_p
      if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
      # Shift the indices to the right to keep also the first token above the threshold
      sorted_indices_to_remove[...,1:] = sorted_indices_to_remove[..., :-1].clone()
      sorted_indices_to_remove[..., 0] = 0

      # scatter sorted tensors to original indexing
      indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
      logits[indices_to_remove] = filter_value
    return logits


  def complete_image(self, text_tokens, batch_size, num_beams, image_tokens=None, top_k = 50, top_p = 0.95):
    config = self.model_args
    if image_tokens is None:
        num_steps = config.total_context_len - text_tokens.shape[1]
    else:
        num_steps = config.total_context_len - text_tokens.shape[1] - image_tokens.shape[1]
    for i in trange(num_steps):
      total_gen = config.text_context_len
      if image_tokens is not None:
        image_tokens = image_tokens.view(batch_size*num_beams, -1)
        total_gen += image_tokens.shape[1]

      with torch.no_grad():
        logits = self.model.forward(
            text_tokens=text_tokens.view(batch_size*num_beams, -1),
            image_tokens=image_tokens,
            attn_mask = torch.ones(text_tokens.size(0), total_gen)
        )[0].cpu() # [batch_size * num_beams, total_gen, image_vocab_size]
      scores = F.log_softmax(logits[:, -1], dim=-1) # [batch_size * num_beams, image_vocab_size]
      top_scores = self.top_k_top_p_filtering(
        logits=scores,
        top_k = top_k,
        top_p=top_p,
        filter_value = -1e6
      )  # [batch_size * num_beams, image_vocab_size]
      # _scores = top_scores.contiguous().view(batch_size, num_beams * top_scores.size(-1)) # (batch_size, num_beams * vocab_size)
      
      # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
      probs = F.softmax(top_scores, dim=-1)
      next_tokens = torch.multinomial(probs, num_samples=num_beams) # (batch_size, num_beams)
    
      next_scores = torch.gather(top_scores, -1, next_tokens) # (batch_size, num_beams)
      
      next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
        
      # gather those with high score for each batch
      next_tokens = torch.gather(next_tokens, -1, next_scores_indices)[:, 0].unsqueeze(1)  # (batch_size, 1)

      if image_tokens is not None:
        image_tokens = torch.cat([image_tokens, next_tokens], dim = -1)
      else:
        image_tokens = next_tokens
    return image_tokens


  def generate(self, texts, num_beams = 1, images=None, image_context_len=100, top_k=50, top_p=0.95):
    # text_tokens = self.tok.encode(cap).ids[:self.textlen - 1] + [self.text_end_id]
    config = self.model_args
    tokens = [self.tokenizer.encode(x * 100).ids[:config.text_context_len - 1] for x in texts]
    tokens = torch.Tensor([[x for _ in range(num_beams)] for x in tokens]).long()
    tokens = tokens.view(num_beams * len(texts), -1)
    eot_tokens = torch.ones(len(tokens), 1) * self.text_end_id
    text_tokens = torch.cat([tokens, eot_tokens], dim=-1).long().to(self.device)
    batch_size = len(texts)

    with torch.no_grad():
      image_tokens = None
      if images is not None and image_context_len > 0:
        image_tokens = self.model.vae._encode_image(images.to(self.device))
        image_tokens = image_tokens[:, :image_context_len]

      complete_image_tokens = self.complete_image(
        text_tokens=text_tokens,
        image_tokens=image_tokens,
        batch_size=batch_size,
        num_beams=num_beams,
        top_k = top_k,
        top_p = top_p
      )

      # now that the tokens are generated we need to pass this to the vae
      gen_image = self.model.vae._decode_ids(image_tokens=complete_image_tokens).permute((0, 2, 3, 1))
    gen_image = (gen_image.numpy() * 255).astype(np.uint8)
    return gen_image


# ---------- model ends


class DallECaptions():
  def __init__(
      self,
      captions_file,
      tokenizer_path,
      keys,
      res = 128,
      text_context_len=64
    ):

    with open(captions_file, "r") as f:
      self.data = json.load(f)
    self.image_keys = list(self.data.keys())
    self.indices = keys

    # image related
    self.t = transforms.Compose([
      transforms.Resize((res, res)),
      transforms.ToTensor()
    ])

    # text related
    self.textlen = text_context_len
    self.tok = Tokenizer.from_file(tokenizer_path)
    self.text_end_id = self.tok.get_vocab()["<|endoftext|>"]
    self.image_end_id = self.tok.get_vocab()["<|endofimage|>"]

    print("Tokenizer loaded with vocab size:", self.tok.get_vocab_size())

  @staticmethod
  def get_split(captions_file, train_split=0.95):
    # we need to get the split index to ensure correct split
    with open(captions_file, "r") as f:
      data = json.load(f)
    image_keys = list(data.keys())
    np.random.shuffle(image_keys)
    train_idx = int(train_split*len(image_keys))
    return image_keys[:train_idx], image_keys[train_idx:]

  def __len__(self):
    return len(self.indices)

  def decode(self, x):
    return [self.tok.decode(y, skip_special_tokens=True) for y in x.tolist()]

  def __getitem__(self, i):
    key = self.indices[i]
    x = self.data[key]
    img = self.t(Image.open(x["path"]).convert('RGB'))
    
    # just force this to be very large, repeat is fine OpenAI DallE does this as well
    cap = x["caption"].lower() * 100
    text_tokens = self.tok.encode(cap).ids[:self.textlen - 1] + [self.text_end_id]
    return {
      "images": img,
      "text_tokens": torch.Tensor(text_tokens).long()
    }


class DallETrainer():
  def __init__(self, train, test, model):
    self.model = model
    self.model_config = model.config
    self.train_dataset=train
    self.test_dataset=test
    self.device="cpu"
    if torch.cuda.is_available():
      self.device=torch.cuda.current_device()
      self.model=torch.nn.DataParallel(self.model).to(self.device)
      print("Model is now CUDA!")

  def save_checkpoint(self, ckpt_path=None):
    raw_model=self.model.module if hasattr(self.model, "module") else self.model
    ckpt_path=ckpt_path if ckpt_path is not None else self.config.ckpt_path
    print(f"Saving Model at {ckpt_path}")
    torch.save(raw_model.state_dict(), ckpt_path)

  def norm_img(self, img):
    img -= img.min()
    img /= img.max()
    return img

  def train(
    self,
    batch_size,
    n_epochs,
    lr,
    folder_path,
    skip_steps,
    weight_decay = 1e-3,
    warmup_perc=0.05,
    test_every=1000,
    test_batch_size=None,
    patience=5,
    gradient_accumulation_steps: int=1,
  ):
    model = self.model
    model_config = self.model_config
    train_data = self.train_dataset
    test_data = self.test_dataset
    epoch_step = len(train_data) // batch_size + int(len(train_data) % batch_size != 0)


    gs = 0                 # global step counter
    train_losses = [-1]    # list with values of training losses at each step
    do_skip = True         # flag for one time skipping steps during training
    min_test_loss = 10000  # any large value for the minimum test loss yer achieved
    patience_counter = 0   # counter for matching the patience
    break_training = False # flag is set when we run out of patience and want to break
                           # training for outer loop as well
    model.train()          # set model to training mode
    _lr = 0                # set rolling _lr for logging dict
    
    # number of steps and warmup for LR scheduling
    total_steps = int(epoch_step * n_epochs)
    warmup_steps = int(warmup_perc * total_steps)
    
    print(f"Warmup/Total: [{warmup_steps}/{total_steps}] | Perc: {warmup_steps/total_steps:.3f} ({warmup_perc})")

    # VAE is not to be optimised, named_parameters() gives more control
    # optim = torch.optim.Adam(
    #   (p for n, p in dalle.named_parameters() if "vae" not in n),
    #   lr=lr
    # )
    optim = configure_optimizers(model, 1., weight_decay, betas = (0.9, 0.99))

    # NOAM scheme from "Attention is all you need"
    lr_lambda_fn = lambda gs: (model_config.n_embd ** -0.5) * min ((gs + 1) ** -0.5, (gs + 1) * (warmup_steps ** -1.5))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda_fn, last_epoch=-1, verbose=False)
    
    if gradient_accumulation_steps > 1:
      eff_batch_size = gradient_accumulation_steps * batch_size
      print(":: Due to presence of gradient_accumulation_steps effective size:", eff_batch_size)

    for epoch in range(n_epochs):
      # ----- train for one complete epoch
      dl = DataLoader(
          dataset=train_data,
          batch_size=batch_size,
          pin_memory=True,    # for CUDA
          shuffle=True,       # of course, my stupid ass didn't do it for first 74 runs
          num_workers=8       # number of workers for parallel loading
      )
      pbar = trange(epoch_step)
      model.zero_grad()

      for d, loop_idx in zip(dl, pbar):
        # don't train if we need to skip some steps but we do not want
        # it to skip for all the future epochs and so we add `do_skip` flag
        if skip_steps and loop_idx < skip_steps and do_skip:
          lr_scheduler.step() # lr also needs to be correct
          continue
        do_skip = False

        # train the model
        d = {k: v.to(self.device) for k,v in d.items()}
        pbar.set_description(f"[TRAIN - {epoch}] GS: {gs}, Loss: {round(train_losses[-1], 5)}")
        _, loss = model(**d, loss = True)
        loss = loss.mean()  # gather from multiple GPUs
        log_dict = {"loss": loss.item()}

        # backprop gradient acc. code:
        # https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3
        loss = loss / gradient_accumulation_steps
        loss.backward()
        if gs and gs % gradient_accumulation_steps == 0:
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          optim.step()

          # this needs to be done before each back prop so that gradients are pointing
          # in the intended minimum. because this needs to be done before each loss.backward() pass
          # I do it at the end of each step because in loop id would behave the same
          # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
          model.zero_grad()
          # ------ update ends

        # # decay based on our progress of training, picked from
        # # https://github.com/karpathy/minGPT/blob/master/mingpt/trainer.py
        # # since we are processing a fixed number of tokens in each step unlike LM-GPT
        # # we can let go of counting the processed tokens and instead just focus on the
        # # global steps processed
        # if gs < warmup_steps:
        #   # linear warmup
        #   lr_mult = gs / max(1, warmup_steps)
        # else:
        #   # cosine decay
        #   progress = (gs - warmup_steps) / max(1, total_steps - warmup_steps)
        #   lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        # _lr = lr * lr_mult
        # for param_group in optim.param_groups:
        #   param_group['lr'] = _lr

        lr_scheduler.step()
        _lr = lr_scheduler.get_last_lr()[0]

        gs += 1
        train_losses.append(loss.item() * gradient_accumulation_steps)
        log_dict.update({"lr": _lr})

        # ----- test loop
        if test_data != None and gs and gs % test_every == 0:
          print(":: Entering Testing Mode")
          if test_batch_size is None:
            test_batch_size = batch_size * 4
          dl=DataLoader(
            dataset=test_data,
            batch_size=test_batch_size, # testing can run larger batches
            pin_memory=True,            # for CUDA
            shuffle=False,              # to ensure we can see the progress being made
            num_workers=8               # number of workers for parallel loading
          )
          model.zero_grad() # remove any gradients from the model
          model.eval()      # convert model to testing mode

          epoch_step_test=len(test_data) // test_batch_size + int(len(test_data) % test_batch_size != 0)
          pbar_test=trange(epoch_step_test)
          test_loss=[]
          for d, e in zip(dl, pbar_test):
            d = {k: v.to(self.device) for k, v in d.items()}
            pbar_test.set_description(f"[TEST - {epoch}]")
            with torch.no_grad():
              _, loss, gen_images = model(**d, loss=True, recons = True)
            loss=loss.mean() # gather from multiple GPUs
            test_loss.append(loss.item())

          # now create samples of the images and
          fig=plt.figure(figsize=(20, 7))
          captions_text = test_data.decode(d["text_tokens"][:10])
          for _i, (i, o, c) in enumerate(zip(d["images"][:10], gen_images[:10], captions_text)):
            i=self.norm_img(i.permute(1, 2, 0).cpu().numpy())
            o=self.norm_img(o.permute(1, 2, 0).cpu().numpy())
            plt.subplot(2, 10, _i + 1)
            plt.imshow(i)
            plt.subplot(2, 10, _i + 10 + 1)
            plt.imshow(o)
            plt.title("\n".join(wrap(c, 20))[:100]) # should be at last
          plt.tight_layout()
          plt.savefig(f"{folder_path}/sample_{gs}.png")
          del fig # delete and save the warning

          test_loss = np.mean(test_loss)
          log_dict.update({"test_loss": test_loss})
          print(":::: Loss:", test_loss)

          if min_test_loss > test_loss:
            print(":: Previous loss was larger, updating value")
            min_test_loss = test_loss
            patience_counter = 0
            self.save_checkpoint(ckpt_path=f"{folder_path}/dalle_{gs}.pt")

          else:
            print(":: Previous loss was smaller, updating value")
            patience_counter += 1

          if patience_counter == patience:
            print(":: Ran out of patience, stopping training")
            break_training = True
            break
          model.train()  # convert model back to training mode

        # ------ testing `if` ends
        if WANDB:
          wandb.log(log_dict)
        else:
          print(log_dict)

        if break_training: break
      # ------ epoch loop ends
    
      if break_training: break
    # ------ training loop ends


if __name__ == "__main__":
  args = ArgumentParser(description= "train DallE transformer model")
  args.add_argument("--vqvae", type=str, default="./vqvae3_128_325_3025_0_ckpt_30600.pt", help="path to the VQVAE_v3 model file")
  args.add_argument("--tokenizer", type=str, default="../tokenizer.json", help="path to the tokenizer")
  args.add_argument("--model_path", type=str, default="./dalle2_128_576_36_12_6/dalle_48000.pt", help="path to pretrained model")
  args.add_argument("--captions", type=str, default="../captions_train.json", help="path to captions file")
  args.add_argument("--text_context_len", type=int, default=128, help="number of tokens in the text")
  args.add_argument("--n_embd", type=int, default=576, help="embedding dimension of the model")
  args.add_argument("--n_layers", type=int, default=36, help="number of attention layers")
  args.add_argument("--n_heads", type=int, default=12, help="number of heads in MHA")
  args.add_argument("--batch_size", type=int, default=6, help="minibatch size")
  args.add_argument("--n_epochs", type=int, default=2, help="number of epochs to train for")
  args.add_argument("--lr", type=int, default=1e-4, help="learning rate")
  args.add_argument("--gas", type=int, default=30, help="gradient accumulation steps")
  args.add_argument("--seed", type=int, default=3, help="seed value")  # 3 = my misha
  args.add_argument("--test_every", type=int, default=4000, help="test every this steps")
  args.add_argument("--patience", type=int, default=2, help="stop training if no improvement in this steps")
  args = args.parse_args()

  # set seed to ensure everything is properly split
  vqvae_arch = Vqvae.infer_details_from_name(args.vqvae)
  set_seed(args.seed)
  folder_path = f"./dalle2_{vqvae_arch.input_res}_{args.n_embd}_" +\
      f"{args.n_layers}_{args.n_heads}_{args.batch_size}"
  print(f":: Will Save data in {folder_path}")
  os.makedirs(folder_path, exist_ok=True)

  train_split = 0.995
  train_keys, test_keys = DallECaptions.get_split(
      captions_file=args.captions, train_split=train_split)
  dallecaptions_train = DallECaptions(
    captions_file=args.captions,
    tokenizer_path=args.tokenizer,
    res=vqvae_arch.input_res,
    keys=train_keys,
    text_context_len=args.text_context_len,
  )
  dallecaptions_test = DallECaptions(
    captions_file=args.captions,
    tokenizer_path=args.tokenizer,
    res=vqvae_arch.input_res,
    keys=test_keys,
    text_context_len=args.text_context_len,
  )
  print("Train Size:", len(dallecaptions_train), "; Test Size:", len(dallecaptions_test))

  # mapping for <res>: <encoded_res>
  resmap = {
    128: 16
  }

  # define the model
  model = Vqvae(args.vqvae)
  transformer_config = TransformerConfig(
    text_context_len=args.text_context_len,
    image_context_len=int(resmap[vqvae_arch.input_res]**2),
    text_vocab_size=dallecaptions_train.tok.get_vocab_size(),
    image_vocab_size=model.num_embeddings,
    n_embd=args.n_embd,
    n_layers=args.n_layers,
    n_heads=args.n_heads
  )
  dalle = DallETransformer(model.get_model(), transformer_config)
  init_weights(dalle) # init weights manually
  print(":: Number of params:", sum(p.numel() for p in dalle.parameters()))

  if args.model_path is not None:
    dalle.load_state_dict(torch.load(args.model_path))

  if WANDB:
    wandb.init(project="dall-e", resume = True)
    wandb.watch(dalle) # watch the model metrics

  # define the trainer
  trainer = DallETrainer(dallecaptions_train, dallecaptions_test, dalle)
  trainer.train(
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
    lr=args.lr,
    skip_steps=48000,
    test_every=args.test_every,
    patience=args.patience,
    gradient_accumulation_steps=args.gas,
    folder_path=folder_path,
    weight_decay=0.01,
    warmup_perc=0.01,
  )
