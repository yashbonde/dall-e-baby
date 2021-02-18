# simple implementation of pure-attention based method as a discrete VAE
# inspired from TransGAN: https://arxiv.org/pdf/2102.07074.pdf
# the code borrowed from: https://github.com/VITA-Group/TransGAN/blob/master/models/ViT_8_8.py
# and also from: # https://github.com/openai/CLIP/blob/main/clip/model.py

import os
import math
import torch
import warnings
from torch import nn, einsum
from argparse import ArgumentParser
from torch.nn import functional as F
from model import QuickGELU, ResidualAttentionBlock
from discrete_vae import DiscreteVAETrainer, set_seed, datasets, transforms

WANDB = os.getenv("WANDB")
if WANDB:
  import wandb


class DSWrapper():
  def __init__(self, res=32, train=False):
    # wrapper for dataset but also has transforms
    self.d = datasets.CIFAR10("./", download=True, train=train)
    if train:
      self.t = transforms.Compose([
          transforms.Resize((res, res)),
          transforms.CenterCrop(res),
          transforms.ToTensor(),
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.RandomVerticalFlip(p=0.5),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])
    else:
      self.t = transforms.Compose([
          transforms.Resize((res, res)),
          transforms.CenterCrop(res),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])

  def __len__(self):
    return len(self.d)

  def __getitem__(self, i):
    img, label = self.d[i]
    return self.t(img)


def truncated_initialisation(tensor, m=0, s=1., a=-2., b=2.):
  # https://github.com/VITA-Group/TransGAN/blob/master/models/ViT_helper.py
  # type: (Tensor, float, float, float, float) -> Tensor

  def norm_cdf(x):
    # computes standard normal cumulative distribution function
    return (1. + math.erf(x / math.sqrt(2.))) / 2.

  if (m < a - 2 * s) or (m > b + 2 * s):
    warnings.warn(
        "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
        "The distribution of values may be incorrect.", stacklevel=2
    )

  with torch.no_grad():
    # values are generated by using a truncated uniform distribution and then
    # using the inverse CDF for normal distribution. Get upper and lower values
    l = norm_cdf((a-m)/s)
    u = norm_cdf((b-m)/s)

    # uniformly fill the tensor values from [l, u] then translate to [2l-1, 2u-1]
    tensor.uniform_(2*l-1, 2*u-1)

    # use inverse cfd transforme for normal distribution
    tensor.erfinv_()

    # transform to proper mean and std
    tensor.mul_(s*math.sqrt(2.)).add_(m)

    # clamp to ensure values
    tensor.clamp_(min=1, max=b)
    return tensor



def pixel_upsample(x, H, W, r = 2):
  B, N, C = x.size()
  assert N == H*W
  x = x.permute(0, 2, 1)
  x = x.view(-1, C, H, W)

  # this function does this rearrangement (r = upscale_factor)
  # [*,C*r2,H,W] -> [*,C,Hxr,Wxr]
  x = nn.PixelShuffle(upscale_factor = r)(x)
  B, C, H, W = x.size()
  x = x.view(-1, C, H*W)
  x = x.permute(0, 2, 1)
  return x, H, W


# unlike nn.PixelShuffle nn.PixelUnshuffle is in 1.9.1 master only
# the TransGAN does this: nn.AvgPool2d(kernel_size=2)(x)
class PixelUnshuffle(nn.Module):
  def forward(self, x, H, W, r = 2):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)

    # reshape [*,C,Hxr,Wxr] --> [*,C*r2,H,W]
    x = x.contiguous().view(-1, C*r*r, H//r, W//r)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0, 2, 1)
    return x, H, W

# init so usage is similar to a function
pixel_downsample = PixelUnshuffle()

class Decoder(nn.Module):
  # similar to Generator
  def __init__(self, n_embd, n_head, bottom_width = 4, depth = 5):
    super().__init__()
    self.bottom_width = bottom_width
    self.n_embd = n_embd

    # different position embedding for different blocks
    self.pos_embd = [
      nn.Parameter(torch.zeros(1, bottom_width ** 2, n_embd)),              # blocks
      nn.Parameter(torch.zeros(1, (bottom_width * 2) ** 2, n_embd // 4)),  # upsample_blocks[0]
      nn.Parameter(torch.zeros(1, (bottom_width * 4) ** 2, n_embd // 16)), # upsample_blocks[1]
    ]
    self.blocks = nn.Sequential(*[
      ResidualAttentionBlock(d_model=n_embd, n_head = n_head)
      for _ in range(depth)
    ])
    self.upsample_blocks = nn.ModuleList([
      nn.Sequential(*[
        ResidualAttentionBlock(d_model = n_embd // 4, n_head = n_head)
        for _ in range(4)
      ]),
      nn.Sequential(*[
        ResidualAttentionBlock(d_model=n_embd // 16, n_head=n_head)
        for _ in range(2)
      ])
    ])

    self.to_rgb = nn.Sequential(
      nn.BatchNorm2d(n_embd),
      QuickGELU(),
      nn.Tanh()
    )

    self.deconv = nn.Conv2d(
      in_channels = n_embd // 16,
      out_channels = 3,
      kernel_size = 1,
      stride = 1,
      padding = 0
    )

  def forward(self, x):
    H, W = self.bottom_width, self.bottom_width
    x = x + self.pos_embd[0].to(x.device)
    x = self.blocks(x)
    for i, blk in enumerate(self.upsample_blocks):
      x, H, W = pixel_upsample(x, H, W, r=2)
      x = x + self.pos_embd[i+1].to(x.device)
      x = blk(x)
    output = self.deconv(x.permute(0, 2, 1).view(-1, self.n_embd // 16, H, W))
    return output


class Encoder(nn.Module):
  # opposite of Decoder module above
  def __init__(self, n_embd, n_head, bottom_width = 4, depth=5):
    super().__init__()
    self.n_embd = n_embd

    self.conv = nn.Conv2d(
      in_channels=3,
      out_channels=n_embd // 16,
      kernel_size=1,
      stride=1,
      padding=0
    )

    # different position embedding for different blocks
    self.pos_embd = [
      nn.Parameter(torch.zeros(1, (bottom_width * 4) ** 2, n_embd // 16)), # downsample_blocks[0]
      nn.Parameter(torch.zeros(1, (bottom_width * 2) ** 2, n_embd // 4)),  # downsample_blocks[1]
      nn.Parameter(torch.zeros(1, bottom_width ** 2, n_embd)),             # blocks
    ]
    self.downsample_blocks = nn.ModuleList([
      nn.Sequential(*[
        ResidualAttentionBlock(d_model=n_embd // 16, n_head=n_head)
        for _ in range(2)
      ]),
      nn.Sequential(*[
        ResidualAttentionBlock(d_model = n_embd // 4, n_head = n_head)
        for _ in range(4)
      ])
    ])
    self.blocks = nn.Sequential(*[
      ResidualAttentionBlock(d_model=n_embd, n_head=n_head, attn_mask=None)
      for _ in range(depth)
    ])

  def forward(self, x):
    B, C, H, W = x.size()
    x = self.conv(x).view(-1, self.n_embd // 16, H*W).permute(0, 2, 1)
    for i, blk in enumerate(self. downsample_blocks):
      x = x + self.pos_embd[i].to(x.device)
      x = blk(x)
      x, H, W = pixel_downsample(x, H, W, r=2)
    x = x + self.pos_embd[-1].to(x.device)
    x = self.blocks(x)
    return x


class TransformerVAE(nn.Module):
  def __init__(self, n_embd, n_head, res=32, quantize=False):
    """If quantize=True an explicit codebook is created
    encoder compresses the input image to a flat list of few tokens and embedding
    if codebook is available then the model also performs lookup else it simply
    passes the encoding to upscaling networks. During training gumbel softmax is
    used where as duting testing hard-softmax is used."""
    super().__init__()
    #  auto determine the botttom size, has relations
    # 32 --> 8, 64 --> 16 128 -> 32 
    self.bottom_width = res // 4
    self.enc = Encoder(n_embd=n_embd, n_head=n_head, bottom_width=self.bottom_width)
    self.dec = Decoder(n_embd=n_embd, n_head=n_head, bottom_width=self.bottom_width)
    self.codebook = nn.Embedding(n_embd, n_embd) if quantize else None

    # initialise the position embeddings with truncated normal
    for tensor in self.enc.pos_embd + self.dec.pos_embd:
      truncated_initialisation(tensor)

    self.encoded_shape = (-1, self.bottom_width**2, n_embd)

  def forward(self, x):
    """to understand more about this forward pass please refer to the VQVAE_v3.forward
    method which has much better documentation."""
    enc_out = self.enc(x)  # [B, (H*W)//16, n_embd]
    if self.codebook is not None:
      if self.training:
        softmax = F.gumbel_softmax(enc_out, tau=1., hard=True, dim=1)
      else:
        softmax = F.softmax(enc_out, dim=1)
        softmax = softmax.scatter_(1, torch.argmax(softmax, dim = 1).unsqueeze(1), 1)
      quantized_inputs = einsum("bdhw,dn->bnhw", softmax, self.codebook.weight)
    else:
      if self.training:
        softmax = F.gumbel_softmax(enc_out, tau=1., hard=True, dim=1)
      else:
        softmax = F.softmax(enc_out, dim=1)
        softmax = softmax.scatter_(1, torch.argmax(softmax, dim=1).unsqueeze(1), 1)
      quantized_inputs = softmax
    
    encoding_ids = torch.argmax(softmax, dim=1).view(enc_out.size(0), -1)
    dec_out = self.dec(quantized_inputs)
    loss = F.mse_loss(dec_out, x)

    # encoding_ids, loss, recons
    return encoding_ids, loss, dec_out


if __name__ == "__main__":
  args = ArgumentParser()
  args.add_argument("--n_embd", type=int, default=128, help="embedding dimension")
  args.add_argument("--n_head", type=int, default=8, help="number of heads in MHA")
  args.add_argument("--res", type=int, default=32, help="resolution of the image")
  args.add_argument("--lr", type=float, default=2e-4, help="learning rate for the model")
  args.add_argument("--test_every", type=int, default=900, help="test model after these steps")
  args.add_argument("--save_every", type=int, default=1800, help="save model after these steps")
  args.add_argument("--batch_size", type=int, default=64, help="minibatch size")
  args.add_argument("--n_epochs", type=int, default=2, help="number of epochs to train for")
  args.add_argument("--seed", type=int, default=3, help="seed value") # 3 = my misha
  args.add_argument(
    "--gradient_accumulation_steps", type=int, default=2.,
    help="perform backward pass after these global steps"
  )
  args = args.parse_args()

  # set seed and ensure everything is properly split
  set_seed(args.seed)
  folder_path = f"./transVAE_{args.res}_{args.n_embd}_{args.batch_size}"
  print(f":: Will Save data in {folder_path}")
  os.makedirs(folder_path, exist_ok=True)

  # define the model
  model = TransformerVAE(n_embd=args.n_embd, n_head=args.n_head, res=args.res)
  print(":: Number of params:", sum(p.numel() for p in model.parameters()))

  if WANDB:
    wandb.init(project="vq-vae")
    wandb.watch(model)  # watch the model metrics

  # define the dataset and goooo
  train = DSWrapper(train=True)
  test = DSWrapper(train=False)
  trainer = DiscreteVAETrainer(model, train, test)
  trainer.train(
    bs = args.batch_size,
    lr = args.lr,
    folder_path=folder_path,
    test_every=args.test_every,
    save_every=args.save_every,
    n_epochs=args.n_epochs,
    skip_steps=None,
    gradient_accumulation_steps=args.gradient_accumulation_steps
  )
