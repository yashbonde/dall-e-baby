# this file has the transformer model inspired from OpenAI CLIP code:
# https://github.com/openai/CLIP/blob/main/clip/model.py

import os
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
from collections import OrderedDict
from argparse import ArgumentParser
from torch.nn import functional as F
from torch.utils.data import DataLoader
from discrete_vae import VQVAE_v3, transforms, set_seed

from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import Block


WANDB = os.getenv("WANDB")
if WANDB:
  import wandb

plt.rcParams.update({
  'font.family': 'barlow',
  'font.size': 10
})


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


class Transformer(nn.Module):
  def __init__(self, total_context_len:int,  n_embd: int, n_layers: int, n_heads: int, attn_mask: torch.Tensor = None):
    super().__init__()
    self.n_embd = n_embd
    self.n_layers = n_layers
    transconfig = GPT2Config(
      n_positions=total_context_len,
      n_ctx=total_context_len,
      n_embd=n_embd,
      n_head=n_heads,
      n_layer=n_layers
    )
    self.attn_mask = attn_mask

    # using the ResidualAttentionBlock from OpenAI was causing problems
    # so using the huggingface GPT2 Block
    self.h = nn.ModuleList([
      Block(transconfig.n_ctx, transconfig, scale=True)
      for _ in range(n_layers)
    ])

  def forward(self, x: torch.Tensor, output_attentions = False):
    hidden_states = x
    for blk in self.h:
      output = blk(
        hidden_states,
        attention_mask=self.attn_mask,
        output_attentions=output_attentions,
      )
      hidden_states = output[-1]
    return [hidden_states]


class DallE(nn.Module):
  def __init__(self, vae, transformer_config):
    super().__init__()
    self.vae = vae

    # transformer
    tconf = transformer_config
    self.tconf = tconf
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
      return mask.unsqueeze(0).unsqueeze(0)

  def forward(self, text_tokens, images, recons = False, loss = False):
    """
    text_tokens: [B, t],
    images: [B, 3, res, res]
    """
    config = self.tconf
    image_tokens = self.vae._encode_image(images) # [B,i]
    text_tokens = text_tokens + config.image_vocab_size # increment because the image tokens are priority
    tokens = torch.cat([text_tokens, image_tokens], dim = -1) # [B,t] + [B,i] = [B,M]
    embed = self.token_embedding(tokens) + self.positional_encoding # [B,M,e]

    # transformer blocks return tuple [hidden_states, present, (attentions, cross_attentions)]
    out = self.transformer(embed)[0] # [B,M,e]
    out = self.image_head(out) # [B,M,vi]
    output = [out]

    if loss:
      logits = out[:, :-1].contiguous().view(-1, config.image_vocab_size)
      
      # note that we do not need to calculate loss for the entire sequence but only for the image tokens
      # so the labels are -100 for text_tokens and image_tokens concatenated
      labels = torch.cat([
        torch.ones_like(text_tokens).long() * -100,
        image_tokens
      ], dim = -1)[:, 1:].contiguous().view(-1)
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


# ---------- model ends


class DallECaptions():
  def __init__(
      self,
      captions_file,
      tokenizer_path,
      train_idx,
      train=False,
      res = 128,
      text_context_len=64
    ):

    with open(captions_file, "r") as f:
      self.data = json.load(f)
    self.image_keys = list(self.data.keys())
    self.indices = self.image_keys[:train_idx] if train else self.image_keys[train_idx:]

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
    return train_idx

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
    self, batch_size, n_epochs, lr, folder_path, skip_steps,
    test_every=500, test_batch_size=None, patience=5,
    gradient_accumulation_steps=1.,
  ):
    model = self.model
    train_data = self.train_dataset
    test_data = self.test_dataset
    epoch_step = len(train_data) // batch_size + int(len(train_data) % batch_size != 0)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    gs = 0                 # global step counter
    train_losses = [-1]    # list with values of training losses at each step
    do_skip = True         # flag for one time skipping steps during training
    min_test_loss = 10000  # any large value for the minimum test loss yer achieved
    patience_counter = 0   # counter for matching the patience
    break_training = False # flag is set when we run out of patience and want to break
                           # training for outer loop as well
    model.train()          # set model to training mode

    for epoch in range(n_epochs):
      # ----- train for one complete epoch
      dl = DataLoader(
          dataset=train_data,
          batch_size=batch_size,
          pin_memory=True,    # for CUDA
          shuffle=True,       # of course, my stupid ass didn't do it for first 74 runs
          num_workers=1       # number of workers for parallel loading
      )
      pbar = trange(epoch_step)

      for d, loop_idx in zip(dl, pbar):
        # don't train if we need to skip some steps but we do not want
        # it to skip for all the future epochs and so we add `do_skip` flag
        if skip_steps and loop_idx < skip_steps and do_skip:
          continue
        do_skip = False

        # train the model
        d = {k: v.to(self.device) for k,v in d.items()}
        pbar.set_description(
            f"[TRAIN - {epoch}] GS: {gs}, Loss: {round(train_losses[-1], 5)}")
        _, loss = model(**d, loss = True)
        loss = loss.mean()  # gather from multiple GPUs
        if WANDB:
          wandb.log({"loss": loss.item()})

        # gradient clipping
        loss = loss / gradient_accumulation_steps
        loss.backward()
        if gs and gs % gradient_accumulation_steps == 0:
          optim.zero_grad()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
          optim.step()
        gs += 1
        train_losses.append(loss.item())

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
            num_workers=1               # number of workers for parallel loading
          )
          model.eval() # convert model to testing mode

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
            plt.title("\n".join(wrap(c, 20))[:100])
            plt.subplot(2, 10, _i + 1)
            plt.imshow(i)
            plt.subplot(2, 10, _i + 10 + 1)
            plt.imshow(o)
          plt.tight_layout()
          plt.savefig(f"{folder_path}/sample_{gs}.png")
          del fig # delete and save the warning

          test_loss = np.mean(test_loss)
          if WANDB:
            wandb.log({"test_loss": test_loss})
          print(":::: Loss:", test_loss)

          if min_test_loss > test_loss:
            print(":: Previous loss was larger, updating value")
            min_test_loss = test_loss
            self.save_checkpoint(ckpt_path=f"{folder_path}/vae_{gs}.pt")

          else:
            print(":: Previous loss was smaller, updating value")
            patience_counter += 1

          if patience_counter == patience:
            print(":: Ran out of patience, stopping training")
            break_training = True
            break
          model.train()  # convert model back to training mode
        
        # ------ testing `if` ends

        if break_training: break
      # ------ epoch loop ends
    
      if break_training: break
    # ------ training loop ends


def init_weights(module):
  if isinstance(module, (nn.Linear, nn.Embedding)):
    module.weight.data.normal_(mean=0.0, std=0.2)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()
  elif isinstance(module, (nn.LayerNorm)):
    module.bias.data.zero_()
    module.weight.data.fill_(1.0)


if __name__ == "__main__":
  args = ArgumentParser(description= "train DallE transformer model")
  args.add_argument("--vqvae", type=str, default="./models/vqvae3_128_325_3025_0_ckpt_28800.pt", help = "path to the VQVAE_v3 model file")
  args.add_argument("--tokenizer", type=str, default="../tokenizer.json", help = "path to the tokenizer")
  args.add_argument("--captions", type=str, default="../captions_train.json", help = "path to captions file")
  args.add_argument("--text_context_len", type=int, default=64, help = "number of tokens in the text")
  args.add_argument("--n_embd", type = int, default = 480, help = "embedding dimension of the model")
  args.add_argument("--n_layers", type = int, default = 12, help = "number of attention layers")
  args.add_argument("--n_heads", type = int, default = 12, help = "number of heads in MHA")
  args.add_argument("--batch_size", type = int, default = 12, help = "minibatch size")
  args.add_argument("--n_epochs", type = int, default = 2, help = "number of epochs to train for")
  args.add_argument("--lr", type = int, default = 1e-5, help = "learning rate")
  args.add_argument("--gas", type = int, default = 1, help = "gradient accumulation steps")
  args.add_argument("--seed", type=int, default=3, help="seed value") # 3 = my misha
  args.add_argument("--test_every", type=int, default=2, help="test every this steps")
  args.add_argument("--patience", type=int, default=5, help="stop training if no improvement in this steps")
  args = args.parse_args()

  # set seed to ensure everything is properly split
  vqvae_arch = Vqvae.infer_details_from_name(args.vqvae)
  set_seed(args.seed)
  folder_path = f"./dalle_{vqvae_arch.input_res}_{args.n_embd}_" +\
      f"{args.n_layers}_{args.n_heads}_{args.batch_size}"
  print(f":: Will Save data in {folder_path}")
  os.makedirs(folder_path, exist_ok=True)

  train_split = 0.3
  train_split = DallECaptions.get_split(
      captions_file=args.captions, train_split=train_split)
  dallecaptions_train = DallECaptions(
    captions_file=args.captions,
    tokenizer_path=args.tokenizer,
    res=vqvae_arch.input_res,
    text_context_len=args.text_context_len,
    train=True,
    train_idx=train_split
  )
  dallecaptions_test = DallECaptions(
    captions_file=args.captions,
    tokenizer_path=args.tokenizer,
    res=vqvae_arch.input_res,
    text_context_len=args.text_context_len,
    train=False,
    train_idx=train_split
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
  dalle = DallE(model.get_model(), transformer_config)
  print(":: Number of params:", sum(p.numel() for p in dalle.parameters()))
  # print(dalle)

  # define the trainer
  trainer = DallETrainer(dallecaptions_train, dallecaptions_test, dalle)
  trainer.train(
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
    lr=args.lr,
    skip_steps=None,
    test_every=args.test_every,
    patience=args.patience,
    gradient_accumulation_steps=args.gas,
    folder_path=folder_path
  )


