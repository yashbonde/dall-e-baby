"""
code to train a discrete VAE
"""
import os
import torch
import random
import argparse
import numpy as np
from glob import glob
from PIL import Image
from uuid import uuid4
from tqdm import trange
from torch import nn, einsum
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

WANDB=os.getenv("WANDB")
if WANDB:
  import wandb

CIFAR_MEAN=[0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
CIFAR_STDS=[0.24703225141799082, 0.24348516474564, 0.26158783926049628]

def set_seed(seed):
  if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Cifar(Dataset):
  def __init__(self, train=True, img_only=False):
    self.c100=datasets.CIFAR100("./data", train=train, download=True)
    self.c10=datasets.CIFAR10("./data", train=train, download=True)
    self.img_only=img_only
    self.transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STDS)
    ])

  def __len__(self):
    return len(self.c100) + len(self.c10)

  def __getitem__(self, i):
    # if i > len(c100) return from c10
    if i >= len(self.c100):
      d=self.c10[i - len(self.c100)]
      cls=self.c10.classes[d[1]]
      img=d[0]
    else:
      d=self.c100[i]
      cls=self.c100.classes[d[1]]
      img=d[0]
    img=self.transform(img)

    if self.img_only:
      return img
    else:
      return img, cls


class BabyDallEDataset(Dataset):
  def __init__(
      self,
      flickr_folder,
      imagenet_folder,
      coco_train_folder,
      coco_val_folder,
      coco_unlabeled,
      train=True,
      train_split=0.98,
      res=102
    ):
    flikr=glob(flickr_folder + "/**/*.jpg")
    imagenet=glob(imagenet_folder + "/**/*.jpg")
    coco_train=glob(coco_train_folder + "/*.jpg")
    coco_val=glob(coco_val_folder + "/*.jpg")
    coco_u=glob(coco_unlabeled+"/*.jpg")
    self.meta={
      "flikr": len(flikr),
      "imagenet": len(imagenet),
      "coco_train": len(coco_train),
      "coco_val": len(coco_val),
      "coco_u":len(coco_u),
      "total": len(flikr)+len(imagenet)+len(coco_train)+len(coco_val)+len(coco_u)
    }
    all_files=flikr+imagenet+coco_train+coco_val+coco_u
    np.random.shuffle(all_files)
    train_idx=int(train_split*len(all_files))
    if train:
        self.files=all_files[:train_idx]
    else:
        self.files=all_files[train_idx:]
    self.t=transforms.Compose([
        transforms.Resize((res, res)),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.Lambda(lambda x: (x-0.5)*2.0),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

  def __repr__(self):
    return "<BabyDallEDataset " + "|".join([f"{k}:{v}" for k,v in self.meta.items()]) + ">"

  def __len__(self):
    return len(self.files)

  def __getitem__(self, i):
    return self.t(Image.open(self.files[i]).convert('RGB'))


# ------ VQVAE model ------- #
# https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
class VectorQuantizer(nn.Module):
  def __init__(
    self,
    num_embeddings: int,
    embedding_dim: int,
    beta: float=0.25
  ):
    super(VectorQuantizer, self).__init__()
    self.K=num_embeddings
    self.D=embedding_dim
    self.beta=beta

    self.embedding=nn.Embedding(self.K, self.D)
    self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

  def forward(self, latents):
    latents=latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
    latents_shape=latents.shape
    flat_latents=latents.view(-1, self.D)  # [BHW x D]

    # Compute L2 distance between latents and embedding weights
    dist=torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
           torch.sum(self.embedding.weight ** 2, dim=1) - \
           2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

    # Get the encoding that has the min distance
    encoding_inds=torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

    # Convert to one-hot encodings
    device=latents.device
    encoding_one_hot=torch.zeros(encoding_inds.size(0), self.K, device=device)
    encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

    # Quantize the latents
    quantized_latents=torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
    quantized_latents=quantized_latents.view(latents_shape)  # [B x H x W x D]

    # Compute the VQ Losses
    commitment_loss=F.mse_loss(quantized_latents.detach(), latents)
    embedding_loss=F.mse_loss(quantized_latents, latents.detach())

    vq_loss=commitment_loss * self.beta + embedding_loss

    # Add the residue back to the latents
    quantized_latents=latents + (quantized_latents - latents).detach()

    return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss, encoding_inds  # [B x D x H x W]

class ResidualLayer(nn.Module):
  def __init__( self, in_channels, out_channels):
    super(ResidualLayer, self).__init__()
    self.resblock=nn.Sequential(
      nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(inplace = True),
      nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
    )

  def forward(self, input):
    return input + self.resblock(input)


class VQVAE_Encoder(nn.Module):
  def __init__(
    self,
    in_channels,
    hidden_dims,
    n_layers,
    embedding_dim
  ):
    super().__init__()
    modules = []
    # Build Encoder
    for h_dim in hidden_dims:
      modules.append(nn.Sequential(nn.Conv2d(
        in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU()
      ))
      in_channels=h_dim

    modules.append(nn.Sequential(nn.Conv2d(
      in_channels, in_channels, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU()
    ))

    for _ in range(n_layers):
      modules.append(ResidualLayer(in_channels, in_channels))
    modules.append(nn.LeakyReLU())

    modules.append(nn.Sequential(
      nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1),
      nn.LeakyReLU())
    )

    self.encoder=nn.Sequential(*modules)

  def forward(self, x):
    return self.encoder(x)


class VQVAE_Decoder(nn.Module):
  def __init__(
    self,
    embedding_dim,
    hidden_dims,
    n_layers,
  ):
    super().__init__()
    modules = []
    modules.append(nn.Sequential(nn.Conv2d(
      embedding_dim,
      hidden_dims[-1],
      kernel_size=3,
      stride=1,
      padding=1),
      nn.LeakyReLU())
    )

    for _ in range(n_layers):
        modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

    modules.append(nn.LeakyReLU())

    hidden_dims.reverse()

    for i in range(len(hidden_dims) - 1):
      modules.append(nn.Sequential(nn.ConvTranspose2d(
        hidden_dims[i],
        hidden_dims[i + 1],
        kernel_size=4,
        stride=2,
        padding=1),
        nn.LeakyReLU()
      ))

    modules.append(nn.Sequential(nn.ConvTranspose2d(
      hidden_dims[-1],
      out_channels=3,
      kernel_size=4,
      stride=2,
      padding=1),
      nn.Tanh()
    ))

    self.decoder=nn.Sequential(*modules)
  
  def forward(self, x):
    return self.decoder(x)


class VQVAE_Encoder_v2(nn.Module):
  def __init__(
      self,
      in_channels,
      hidden_dims,
      n_layers,
      embedding_dim
  ):
    super().__init__()
    modules = []
    # Build Encoder
    for h_dim in hidden_dims:
      modules.append(nn.Sequential(
        nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(h_dim),
        nn.LeakyReLU()
      ))
      in_channels = h_dim
      modules.append(ResidualLayer(in_channels, in_channels))
      modules.append(nn.LeakyReLU())

    modules.append(nn.Sequential(
        nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1),
        # nn.LeakyReLU()
    ))

    self.encoder = nn.Sequential(*modules)

  def forward(self, x):
    return self.encoder(x)


class VQVAE_Decoder_v2(nn.Module):
  def __init__(
      self,
      embedding_dim,
      hidden_dims,
      n_layers,
  ):
    super().__init__()
    modules = []
    modules.append(nn.Sequential(nn.Conv2d(
        embedding_dim,
        hidden_dims[-1],
        kernel_size=3,
        stride=1,
        padding=1),
        nn.LeakyReLU())
    )
    hidden_dims.reverse()
    hidden_dims = hidden_dims + [3] # add the last layer in it
    print(hidden_dims)

    for i in range(len(hidden_dims)-1):
      modules.append(ResidualLayer(hidden_dims[i], hidden_dims[i]))
      modules.append(nn.LeakyReLU())
      modules.append(nn.Sequential(nn.ConvTranspose2d(
        hidden_dims[i],
        hidden_dims[i + 1],
        kernel_size=4,
        stride=2,
        padding=1),
        nn.LeakyReLU()
      ))
    modules.append(nn.Tanh())
    self.decoder = nn.Sequential(*modules)

  def forward(self, x):
    return self.decoder(x)


class VQVAE(nn.Module):
  def __init__(
      self,
      in_channels: int,
      embedding_dim: int,
      num_embeddings: int,
      hidden_dims=[128, 256],
      n_layers=6,
      beta: float=0.25,
      img_size: int=64,
      v2=True
    ):
    super().__init__()

    self.embedding_dim=embedding_dim
    self.num_embeddings=num_embeddings
    self.img_size=img_size
    self.beta=beta

    ENC = VQVAE_Encoder_v2 if v2 else VQVAE_Encoder
    DEC = VQVAE_Decoder_v2 if v2 else VQVAE_Decoder

    self.encoder = ENC(
      in_channels=in_channels,
      hidden_dims=hidden_dims,
      n_layers=n_layers,
      embedding_dim=embedding_dim
    )
    self.vq_layer=VectorQuantizer(
      num_embeddings=num_embeddings,
      embedding_dim=embedding_dim,
      beta=self.beta
    )
    self.decoder = DEC(
      embedding_dim=embedding_dim,
      hidden_dims=hidden_dims,
      n_layers=n_layers,
    )

  def forward(self, input, v=False):
    encoding=self.encoder(input)
    if v: print("encoding:", encoding.size())
    quantized_inputs, vq_loss, encoding_inds=self.vq_layer(encoding)
    if v: print("quantized_inputs:", quantized_inputs.size())
    recons=self.decoder(quantized_inputs)
    if v: print("recons:", recons.size())
    recons_loss=F.mse_loss(recons, input)
    # recons_loss = -F.kl_div(input, recons)
    if v: print("recons_loss:", recons_loss)
    loss=recons_loss + vq_loss
    if v: print("loss:", loss)
    return encoding_inds, loss, recons


class VQVAE_v3(nn.Module):
  def __init__(
      self,
      in_channels: int,
      embedding_dim: int,
      num_embeddings: int,
      hidden_dims=[128, 256],
      n_layers=6,
      beta: float=0.25,
      img_size: int=64,
      v2=True
    ):
    super().__init__()

    self.embedding_dim=embedding_dim
    self.num_embeddings=num_embeddings
    self.img_size=img_size
    self.beta=beta

    ENC = VQVAE_Encoder_v2 if v2 else VQVAE_Encoder
    DEC = VQVAE_Decoder_v2 if v2 else VQVAE_Decoder

    self.encoder = ENC(
      in_channels=in_channels,
      hidden_dims=hidden_dims,
      n_layers=n_layers,
      embedding_dim=embedding_dim
    )
    # self.vq_layer=VectorQuantizer(
    #   num_embeddings=num_embeddings,
    #   embedding_dim=embedding_dim,
    #   beta=self.beta
    # )
    self.codebook = nn.Embedding(num_embeddings, embedding_dim)
    self.decoder = DEC(
      embedding_dim=embedding_dim,
      hidden_dims=hidden_dims,
      n_layers=n_layers,
    )

  def forward(self, input, v=False):
    encoding=self.encoder(input)
    if v: print("encoding:", encoding.size())
    # quantized_inputs, vq_loss, encoding_inds=self.vq_layer(encoding)

    # if we are training then we use the soft-gumbel, during eval we use hard-gumbel
    # is_training = self.training()
    # if is_training:
    #   softmax = F.gumbel_softmax(encoding, tau = 1., dim = 1)
    # else:
    #   softmax = F.gumbel_softmax(encoding, tau = 0.75, hard = True, dim = 1)
    softmax = F.gumbel_softmax(encoding, tau = 1., hard = True, dim = 1)
    quantized_inputs = einsum("bnhw,nd->bdhw", softmax, self.codebook.weight)
    encoding_inds = torch.argmax(softmax, dim = 1).view(encoding.size(0), -1)

    vq_loss = F.mse_loss(quantized_inputs.detach(), encoding) + \
      0.25 * F.mse_loss(quantized_inputs, encoding.detach())

    if v: print("quantized_inputs:", quantized_inputs.size())
    recons=self.decoder(quantized_inputs)
    if v: print("recons:", recons.size())
    recons_loss=F.mse_loss(recons, input)
    # recons_loss = -F.kl_div(input, recons)
    if v: print("recons_loss:", recons_loss)
    loss=recons_loss + vq_loss
    if v: print("loss:", loss)
    return encoding_inds, loss, recons

# -------- simple discrete VAE
class DiscreteVAE(nn.Module):
  def __init__(
      self,
      hdim=128,
      num_layers=6,
      vocab=1024,
      embedding_dim=256
    ):
    super().__init__()
    encoder_layers=[]
    decoder_layers=[]
    for i in range(num_layers):
      # now add encoder layer
      enc_layer=nn.Conv2d(
        in_channels=hdim if i > 0 else 3,
        out_channels=hdim,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
      )
      encoder_layers.extend([enc_layer, nn.ReLU()])

      # now add decoder layer
      dec_layer=nn.ConvTranspose2d(
        in_channels=embedding_dim if i == 0 else hdim,
        out_channels=hdim,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
      )
      decoder_layers.extend([dec_layer, nn.ReLU()])

    encoder_layers.append(nn.Conv2d(
      in_channels=hdim,
      out_channels=vocab,
      kernel_size=3
    ))
    decoder_layers.append(nn.ConvTranspose2d(
      in_channels=hdim,
      out_channels=3,
      kernel_size=3,
    ))

    self.encoder=nn.Sequential(*encoder_layers)
    self.codebook=nn.Embedding(vocab, embedding_dim)
    self.decoder=nn.Sequential(*decoder_layers)

  def forward(self, x):
    enc=self.encoder(x)
    soft_one_hot=F.gumbel_softmax(enc, tau=1.)
    hid_tokens=torch.einsum("bnwh,nd->bdwh", soft_one_hot, self.codebook.weight)
    out=self.decoder(hid_tokens)
    return out

# ----------- Residual VAE ----------- #
class EncoderBlock(nn.Module):
  def __init__(self, hidden_dim, out_channels, act="relu", bn=True):
    super().__init__()
    self.resblock=nn.Sequential(
      nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1, bias=False),
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=1, bias=False)
    )
    self.bn=nn.BatchNorm2d(hidden_dim) if bn else None
    self.down_conv=nn.Conv2d(hidden_dim, out_channels, kernel_size=3, stride=2)
    if act=="relu":
      self.act=nn.LeakyReLU(inplace=True)
    elif act=="tanh":
      self.act=nn.Tanh()
    else:
      self.act=None

  def forward(self, x):
    # res -> bn -> relu -> conv
    out=x+self.resblock(x)
    if self.bn is not None:
      out=self.bn(out)
    if self.act is not None:
      out=self.act(out)
    out=self.down_conv(out)
    return out

class DecoderBlock(nn.Module):
  def __init__(self, hidden_dim, out_channels, act="relu", bn=False):
    super().__init__()
    self.resblock=nn.Sequential(
      nn.ConvTranspose2d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1, bias=False),
      nn.LeakyReLU(inplace=True),
      nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=1, bias=False)
    )
    self.bn=nn.BatchNorm2d(hidden_dim) if bn else None
    self.up_conv=nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2)
    if act == "relu":
      self.act=nn.LeakyReLU(inplace=True)
    elif act == "tanh":
      self.act=nn.Tanh()
    else:
      self.act=None

  def forward(self, x):
    # res -> bn -> relu -> conv
    out=x+self.resblock(x)
    if self.bn is not None:
      out=self.bn(out)
    if self.act is not None:
      out=self.act(out)
    out=self.up_conv(out)
    return out

class DiscreteResidualVAE(nn.Module):
  def __init__(self, hidden_dim, n_layers, num_embeds, codebook_dim, in_channels=3, temp=1.0):
    super().__init__()
    self.temp=temp
    encoder=[]
    decoder=[]
    for i in range(n_layers-1):
      encoder.append(EncoderBlock(
        hidden_dim=in_channels if i == 0 else hidden_dim,
        out_channels=hidden_dim,
        act="relu",
        bn=True
      ))
      decoder.append(DecoderBlock(
        hidden_dim=codebook_dim if i == 0 else hidden_dim,
        out_channels=hidden_dim,
        act="relu",
        bn=True
      ))

    # add last encoder and decoder
    # encoder.append(EncoderBlock(
    #   hidden_dim=hidden_dim,
    #   out_channels=num_embeds,
    #   act=None,
    #   bn=True
    # ))
    # decoder.append(DecoderBlock(
    #   hidden_dim=hidden_dim,
    #   out_channels=in_channels,
    #   act="tanh",
    #   bn=True
    # ))

    encoder.append(nn.Conv2d(
      in_channels=hidden_dim,
      out_channels=num_embeds,
      kernel_size=4,
      stride=2,
      bias=False
    ))
    decoder.append(nn.ConvTranspose2d(
      in_channels=hidden_dim,
      out_channels=in_channels,
      kernel_size=4,
      stride=2
    ))
    decoder.append(nn.Tanh())

    self.encoder=nn.Sequential(*encoder)
    self.codebook=nn.Embedding(num_embeds, codebook_dim)
    self.decoder=nn.Sequential(*decoder)

  def forward(self, x):
    out=self.encoder(x)
    # the output from encoder has shape [bnhw] and so we must perform
    # softmax over n (number of embeddings/vocab size) and so dim=1
    # gumbel_softmax learns finer things compared to softmax
    out=F.gumbel_softmax(out, tau=self.temp, dim=1)

    # if we just use softmax the images become flat, you can see where
    # it is attending and what are the important locations, however
    # there are no colours and the image is basically black and white
    # unnormed gradient?
    # out=out / self.temp
    # out=F.softmax(out, dim=1)
    out=einsum("bnhw,nd->bdhw", out, self.codebook.weight)
    out=self.decoder(out)
    loss=F.mse_loss(x, out)
    return None, loss, out


# ------- a different take on VQVAE
class Residualv2(nn.Module):
  def __init__(self, hid):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(hid, hid, 3, stride=1, padding = 1),
      nn.LeakyReLU(inplace = True),
      nn.Conv2d(hid, hid, 3, stride=1, padding = 1),
    )

  def forward(self, x):
    out = self.block(x)
    return x + out
  
class Encoderv2(nn.Module):
  def __init__(self, hid, out_dim, in_channels = 3):
    super().__init__()
    self.c1 = nn.Conv2d(in_channels, hid, kernel_size=4, stride=2, padding = 1)
    self.c2 = nn.Conv2d(hid, hid, kernel_size=4, stride=2, padding = 1)
    self.c3 = nn.Conv2d(hid, hid, kernel_size=4, stride=2, padding = 0)
    self.r1 = Residualv2(hid)
    self.r2 = Residualv2(hid)
    self.mp = nn.Conv2d(hid, out_dim, kernel_size=3, stride=1, padding=1)
  def forward(self, x):
    out = F.leaky_relu(self.c1(x), inplace=True)
    out = F.leaky_relu(self.c2(out), inplace = True)
    out = F.leaky_relu(self.c3(out), inplace = True)
    out = self.r1(out)
    out = self.r2(out)
    out = self.mp(out)
    return out
    
class Decoderv2(nn.Module):
  def __init__(self, hid, in_channels, out_dim = 3):
    super().__init__()
    self.mp = nn.Conv2d(in_channels, hid, kernel_size=3, stride=1, padding=1)
    self.r1 = Residualv2(hid)
    self.r2 = Residualv2(hid)
    self.c1 = nn.ConvTranspose2d(hid, hid, kernel_size=4, stride=2, padding = 0)
    self.c2 = nn.ConvTranspose2d(hid, hid, kernel_size=4, stride=2, padding = 1)
    self.c3 = nn.ConvTranspose2d(hid, out_dim, kernel_size=4, stride=2, padding=1)
  def forward(self, x):
    out = F.leaky_relu(self.mp(x), inplace=True)
    out = self.r1(out)
    out = self.r2(out)
    out = F.leaky_relu(self.c1(out), inplace=True)
    out = F.leaky_relu(self.c2(out), inplace=True)
    out = torch.tanh(self.c3(out))
    return out
  
class VQVAEv2(nn.Module):
  def __init__(self, hid, vocab_size):
    super().__init__()
    self.enc = Encoderv2(hid, vocab_size)
    self.vocab = nn.Embedding(vocab_size, hid)
    self.dec = Decoderv2(hid, hid)
  def forward(self, x):
    out = self.enc(x)

    # first we perform the gumbel softmax for the reparameterisation trick
    softmax = F.gumbel_softmax(out, tau=1., dim = 1)
    embedding = einsum("bdhw,dn->bnhw", softmax, self.vocab.weight)
    
    out = self.dec(embedding)
    recon_loss = F.mse_loss(x, out)
    return softmax, recon_loss, out
  

# ------- trainer
class DiscreteVAETrainer:
  def __init__(self, model, train, test=None):
    self.model=model
    # self.train_dataset=Cifar(train=True, img_only=True)  # define the train dataset
    # self.test_dataset=Cifar(train=False, img_only=True)  # define the test dataset
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

  def norm_img(self,img):
    img -= img.min()
    img /= img.max()
    return img

  def train(self, bs, n_epochs, lr, unk_id, save_every=500, test_every=500):
    model=self.model
    train_data=self.train_dataset
    test_data=self.test_dataset
    epoch_step=len(train_data) // bs + int(len(train_data) % bs != 0)
    save_every=min([save_every, epoch_step])
    num_steps=epoch_step * n_epochs
    prev_loss=10000
    no_improve_step=0
    optim=torch.optim.Adam(model.parameters(), lr=lr)

    gs=0
    train_losses=[-1]
    model.train()
    for epoch in range(n_epochs):
      # ----- train for one complete epoch
      dl=DataLoader(
        dataset=train_data,
        batch_size=bs,
        pin_memory=True
      )
      pbar=trange(epoch_step)
      v=False
      for d, e in zip(dl, pbar):
        d=d.to(self.device)
        pbar.set_description(f"[TRAIN - {epoch}] GS: {gs}, Loss: {round(train_losses[-1], 5)}")
        _, loss, _=model(d)
        loss=loss.mean() # gather from multiple GPUs
        if WANDB:
          wandb.log({"loss": loss.item()})

        # gradient clipping
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        optim.step()
        gs += 1
        train_losses.append(loss.item())

        # ----- test condition
        if test_data != None and gs and gs % test_every == 0:
          print(":: Entering Testing Mode")
          dl=DataLoader(
            dataset=test_data,
            batch_size=bs,
            pin_memory=True
          )
          model=model.eval() # convert model to testing mode
          epoch_step_test=len(test_data) // bs + int(len(test_data) % bs != 0)
          pbar_test=trange(epoch_step_test)
          test_loss=[]
          for d, e in zip(dl, pbar_test):
            d=d.to(self.device)
            pbar_test.set_description(f"[TEST - {epoch}]")
            with torch.no_grad():
              _, loss, output_img=model(d)
            loss=loss.mean() # gather from multiple GPUs
            test_loss.append(loss.item())

          # now create samples of the images and
          fig=plt.figure(figsize=(20, 7))
          for _i,(i,o) in enumerate(zip(d[:10], output_img[:10])):
            i=self.norm_img(i.permute(1, 2, 0).cpu().numpy())
            o=self.norm_img(o.permute(1, 2, 0).cpu().numpy())
            plt.subplot(2, 10, _i + 1)
            plt.imshow(i)
            plt.subplot(2, 10, _i + 10 + 1)
            plt.imshow(o)
          plt.tight_layout()
          plt.savefig(f"./sample_{gs}.png")

          test_loss=np.mean(test_loss)
          if WANDB:
            wandb.log({"test_loss": test_loss})

          print(":::: Loss:", test_loss)
          if prev_loss > test_loss:
            print(":: Improvement over previous model")
            self.save_checkpoint(ckpt_path=f"models/vae_{unk_id}{gs}.pt")
            no_improve_step=0
            prev_loss=test_loss
          else:
            no_improve_step += 1
            print(":: No Improvement for:", no_improve_step, "steps!")

          if no_improve_step == 3:
            print("::: No Improvement in 3 epochs, break training")
            break
          model=model.train()  # convert model back to training mode

    print("EndSave")
    self.save_checkpoint(ckpt_path=f"models/vae_{unk_id}end.pt")
    with open("models/vae_loss.txt", "w") as f:
      f.write("\n".join([str(x) for x in train_losses]))


def init_weights(module):
  if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.ConvTranspose2d)):
    module.weight.data.normal_(mean=0.0, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()
  elif isinstance(module, (nn.LayerNorm)):
    module.bias.data.zero_()
    module.weight.data.fill_(1.0)


if __name__ == "__main__":
  args=argparse.ArgumentParser(description="script to train the VectorQuantised-VAE")
  args.add_argument("--embedding_dim", type=int, default=128, help="embedding dimension to use")
  args.add_argument("--res", type=int, default=64, help="resolution of the image")
  args.add_argument("--num_embeddings", type=int, default=512, help="number of embedding values to use")
  args.add_argument("--n_layers", type=int, default=4, help="number of layers in the model")
  args.add_argument("--lr", type=float, default=2e-4, help="learning rate for the model")
  args.add_argument("--test_every", type=int, default=300, help="test model after these steps")
  args.add_argument("--batch_size", type=int, default=128, help="minibatch size")
  args.add_argument("--n_epochs", type=int, default=30, help="number of epochs to train for")
  args.add_argument("--model", type=str, default="vqvae3",
                    choices=["res", "vqvae", "disvae", "vqvae2", "vqvae3"],
    help="model architecture to use")
  args.add_argument("--dataset", type=str, default="flikr",
    choices=["flikr", "cifar"], help="dataset version to use")
  args=args.parse_args()

  # set seed to ensure everything is properly split
  set_seed(4)

  if args.model == "vqvae":
    print(":: Building VQVAE")
    model=VQVAE(
      in_channels=3,
      embedding_dim=args.embedding_dim*4,
      num_embeddings=args.num_embeddings,
      img_size=args.res,
      hidden_dims=[args.embedding_dim, args.embedding_dim * 2],
    )
  elif args.model == "disvae":
    print(":: Building DiscreteVAE")
    model=DiscreteVAE(
      hdim=args.embedding_dim,
      num_layers=args.n_layers, # 6
      num_tokens=args.num_embeddings,
      embedding_dim=args.embedding_dim
    )
  elif args.model == "res":
    print(":: Building DiscreteResidualVAE")
    model=DiscreteResidualVAE(
      hidden_dim=args.embedding_dim,
      n_layers=args.n_layers,
      num_embeds=args.num_embeddings,
      codebook_dim=args.embedding_dim*2
    )
  elif args.model == "vqvae2":
    model=VQVAEv2(
      hid=args.embedding_dim,
      vocab_size=args.num_embeddings
    )
  elif args.model == "vqvae3":
    model=VQVAE_v3(
      in_channels=3,
      embedding_dim=args.embedding_dim*4,
      num_embeddings=args.num_embeddings,
      img_size=args.res,
      hidden_dims=[args.embedding_dim, args.embedding_dim * 2],
    )
  else:
    raise ValueError("model should be one of `res`, `disvae`, `vqvae`")

  model.apply(init_weights) # initialise weights

  cifar=False
  if args.dataset == "flikr":
    # likr_folder, imagenet_folder, coco_folder
    train=BabyDallEDataset(
      flickr_folder="../flickr30k_images/",
      imagenet_folder="../ImageNet-Datasets-Downloader/data/",
      coco_train_folder="../train2017/",
      coco_val_folder="../val2017/",
      coco_unlabeled ="../unlabeled2017/",
      res=args.res,
      train=True
    )
    test=BabyDallEDataset(
      flickr_folder="../flickr30k_images/",
      imagenet_folder="../ImageNet-Datasets-Downloader/data/",
      coco_train_folder="../train2017/",
      coco_val_folder="../val2017/",
      coco_unlabeled ="../unlabeled2017/",
      res=args.res,
      train=False
    )
    print(":: Loaded BabyDallEDataset", len(train), len(test))
    if len(train) == 0:
        cifar=True

  if args.dataset == "cifar" or cifar:
    print(":: Loading cifar dataset")
    train=Cifar(True, True)
    test=Cifar(False, True)

  local_run=""
  if WANDB:
    wandb.init(project="vq-vae")
    wandb.watch(model) # watch the model metrics
    local_run=str(uuid4())[:8] + "_"
    print(":: Local Run ID:", local_run)
  print(":: Number of params:", sum(p.numel() for p in model.parameters()))
  trainer=DiscreteVAETrainer(model, train, test)
  trainer.train(
    bs=args.batch_size,
    n_epochs=args.n_epochs,
    lr=args.lr,
    unk_id=local_run,
    test_every=args.test_every
  )
