# this file has the transformer model inspired from OpenAI CLIP code:
# https://github.com/openai/CLIP/blob/main/clip/model.py

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from discrete_vae import VQVAE_v3

class Vqvae:
  """
  Wrapper for model in discrete_vae, automatically infers the architecture from model path
  """
  def __init__(self, model_path):
    # model_path looks like this path/to/vqvae3_128_325_3025_0_64
    self.model_path=model_path
    attrs=model_path.split("/")[-1].split("_")
    self.in_channels=3                    # number of channels in the image (def=3)
    self.input_res=int(attrs[1])          # input resolution of the image
    self.embedding_dim=int(attrs[2])*3    # embedding dimension for the latent space
    self.num_embeddings=int(attrs[3])     # number of embeddings in the codebook
    self.add_residual=bool(int(attrs[4])) # to use the model with residual connection or not
    dim = int(attrs[2])
    self.hidden_dims=[dim, int(1.5 * dim), dim * 2] # hidden dimensions for different layers

  def get_model(self):
    model = VQVAE_v3(
        in_channels=self.in_channels,
        embedding_dim=self.embedding_dim,
        num_embeddings=self.num_embeddings,
        hidden_dims=self.hidden_dims,
        add_residual=self.add_residual,
    )
    model.load_state_dict(torch.load(self.model_path))
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



class QuickGELU(nn.Module):
  def forward(self, x: torch.Tensor):
    return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
  def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
    super().__init__()

    self.attn = nn.MultiheadAttention(d_model, n_head)
    self.ln_1 = nn.LayerNorm(d_model)
    self.mlp = nn.Sequential(OrderedDict([
      ("c_fc", nn.Linear(d_model, d_model * 4)),
      ("gelu", QuickGELU()),
      ("c_proj", nn.Linear(d_model * 4, d_model))
    ]))
    self.ln_2 = nn.LayerNorm(d_model)
    self.attn_mask = attn_mask

  def attention(self, x: torch.Tensor):
    self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
    return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

  def forward(self, x: torch.Tensor):
    x = x + self.attention(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

class Transformer(nn.Module):
  def __init__(self, n_embd: int, n_layers: int, n_heads: int, attn_mask: torch.Tensor = None):
    super().__init__()
    self.n_embd = n_embd
    self.n_layers = n_layers
    self.resblocks = nn.Sequential(*[
      ResidualAttentionBlock(d_model=n_embd, n_head=n_heads, attn_mask=attn_mask)
      for _ in range(n_layers)
    ])

  def forward(self, x: torch.Tensor):
    return self.resblocks(x)


class DallE(nn.Module):
  def __init__(self, vae, transformer_config):
    super().__init__()
    self.vae = vae.get_model()

    # transformer
    tconf = transformer_config
    self.tconf = tconf
    self.context_length=tconf.text_context_len + tconf.image_context_len

    # this does not need to be a nn.Embedding because the full length will always be used
    self.positional_encoding = nn.Parameter(torch.empty(self.context_length, tconf.n_embd))
    self.token_embedding = nn.Embedding(tconf.text_vocab_size + tconf.image_vocab_size, tconf.n_embd)
    self.transformer = Transformer(
      n_embd=tconf.n_embd,
      n_layers=tconf.n_layers,
      n_heads=tconf.n_heads,
      attn_mask=self.build_attention_mask()
    )
    self.image_head = nn.Sequential(
      nn.LayerNorm(tconf.n_embd),
      nn.Linear(tconf.n_embd, tconf.image_vocab_size)
    )

  def build_attention_mask(self):
      # lazily create causal attention mask, with full attention between the vision tokens
      # pytorch uses additive attention mask; fill with -inf
      mask = torch.empty(self.context_length, self.context_length)
      mask.fill_(-1e6) # some shit happens with float("-inf")
      mask.triu_(1)    # zero out the lower diagonal
      return mask #.unsqueeze(0).unsqueeze(0)

  def forward(self, text_tokens, images, loss = False):
    """
    text_tokens: [B, t],
    images: [B, 3, res, res]
    """
    config = self.tconf
    image_tokens = self.vae._encode_image(images) # [B,i]
    tokens = torch.cat([text_tokens, image_tokens], dim = -1) # [B,t] + [B,i] = [B,M]
    embed = self.token_embedding(tokens) + self.positional_encoding # [B,M,e]
    
    # permutations required for nn.MultiheadAttention
    embed = embed.permute((1, 0, 2))
    out = self.transformer(embed) # [B,M,e]
    out = out.permute((1, 0, 2))
    out = self.image_head(out) # [B,M,vi]

    output = [out]

    if loss is not None:
      logits = out[:, :-1].contiguous().view(-1, config.image_vocab_size)
      
      # note that we do not need to calculate loss for the entire sequence but only for the image tokens
      # so the labels are -100 for text_tokens and image_tokens concatenated
      labels = torch.cat([
        torch.ones_like(text_tokens).long() * -100,
        image_tokens
      ], dim = -1)[:, 1:].contiguous().view(-1)
      loss = F.cross_entropy(logits, labels)
      output = [out, loss]

    return output

