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
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

WANDB = os.getenv("WANDB")
if WANDB:
  import wandb

CIFAR_MEAN = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
CIFAR_STDS = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

def set_seed(seed):
  if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Cifar(Dataset):
  def __init__(self, train = True, img_only = False):
    self.c100 = datasets.CIFAR100("./data", train=train, download=True)
    self.c10 = datasets.CIFAR10("./data", train=train, download=True)
    self.img_only = img_only
    self.transform = transforms.Compose([
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
      d = self.c10[i - len(self.c100)]
      cls = self.c10.classes[d[1]]
      img = d[0]
    else:
      d = self.c100[i]
      cls = self.c100.classes[d[1]]
      img = d[0]
    img = self.transform(img)
    
    if self.img_only:
      return img
    else:
      return img, cls


class FlikrDataset(Dataset):
  def __init__(self, flikr_folder, imagenet_folder, train = True, train_split=0.98, res=102):
    all_files = glob(flikr_folder + "/**/*.jpg")
    all_files += glob(imagenet_folder + "/**/*.jpg")
    np.random.shuffle(all_files)
    train_idx = int(train_split*len(all_files))
    if train:
        self.files = all_files[:train_idx]
    else:
        self.files = all_files[train_idx:]
    self.t = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STDS)
    ])
    
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
    beta: float = 0.25
  ):
    super(VectorQuantizer, self).__init__()
    self.K = num_embeddings
    self.D = embedding_dim
    self.beta = beta

    self.embedding = nn.Embedding(self.K, self.D)
    self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

  def forward(self, latents):
    latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
    latents_shape = latents.shape
    flat_latents = latents.view(-1, self.D)  # [BHW x D]

    # Compute L2 distance between latents and embedding weights
    dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
           torch.sum(self.embedding.weight ** 2, dim=1) - \
           2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

    # Get the encoding that has the min distance
    encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

    # Convert to one-hot encodings
    device = latents.device
    encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
    encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

    # Quantize the latents
    quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
    quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

    # Compute the VQ Losses
    commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
    embedding_loss = F.mse_loss(quantized_latents, latents.detach())

    vq_loss = commitment_loss * self.beta + embedding_loss

    # Add the residue back to the latents
    quantized_latents = latents + (quantized_latents - latents).detach()

    return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss, encoding_inds  # [B x D x H x W]

class ResidualLayer(nn.Module):
  def __init__( self, in_channels, out_channels):
    super(ResidualLayer, self).__init__()
    self.resblock = nn.Sequential(
      nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1, bias=False),
      nn.ReLU(True),
      nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
    )

  def forward(self, input):
    return input + self.resblock(input)


class VQVAE(nn.Module):
  def __init__(self,
              in_channels: int,
              embedding_dim: int,
              num_embeddings: int,
              hidden_dims = None,
              beta: float = 0.25,
              img_size: int = 64):
    super().__init__()

    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    self.img_size = img_size
    self.beta = beta

    modules = []
    if hidden_dims is None:
        hidden_dims = [128, 256]

    # Build Encoder
    for h_dim in hidden_dims:
        modules.append(nn.Sequential(nn.Conv2d(
          in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
          nn.LeakyReLU()
        ))
        in_channels = h_dim

    modules.append(nn.Sequential(nn.Conv2d(
      in_channels, in_channels, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU()
    ))

    for _ in range(6):
        modules.append(ResidualLayer(in_channels, in_channels))
    modules.append(nn.LeakyReLU())

    modules.append(nn.Sequential(
      nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1),
      nn.LeakyReLU())
    )

    self.encoder = nn.Sequential(*modules)

    self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta)

    # Build Decoder
    modules = []
    modules.append(nn.Sequential(nn.Conv2d(
      embedding_dim,
      hidden_dims[-1],
      kernel_size=3,
      stride=1,
      padding=1),
      nn.LeakyReLU())
    )

    for _ in range(6):
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

    self.decoder = nn.Sequential(*modules)

  def forward(self, input, v = False):
    encoding = self.encoder(input)
    if v: print("encoding:", encoding.size())
    quantized_inputs, vq_loss, encoding_inds = self.vq_layer(encoding)
    if v: print("quantized_inputs:", quantized_inputs.size())
    recons = self.decoder(quantized_inputs)
    if v: print("recons:", recons.size())
    recons_loss = F.mse_loss(recons, input)
    if v: print("recons_loss:", recons_loss)
    loss = recons_loss + vq_loss
    if v: print("loss:", loss)
    return recons, loss, encoding_inds


class DiscreteVAE(nn.Module):
  def __init__(
      self,
      hdim = 128,
      num_layers = 6,
      vocab = 1024,
      embedding_dim = 256
    ):
    super().__init__()
    encoder_layers = []
    decoder_layers = []
    for i in range(num_layers):
      # now add encoder layer
      enc_layer = nn.Conv2d(
        in_channels = hdim if i > 0 else 3,
        out_channels = hdim,
        kernel_size = 3,
        stride = 1,
        padding = 0,
        dilation = 1,
      )
      encoder_layers.extend([enc_layer, nn.ReLU()])
      
      # now add decoder layer
      dec_layer = nn.ConvTranspose2d(
        in_channels = embedding_dim if i == 0 else hdim,
        out_channels = hdim,
        kernel_size = 3,
        stride = 1,
        padding = 0,
        dilation = 1,
      )
      decoder_layers.extend([dec_layer, nn.ReLU()])
        
    encoder_layers.append(nn.Conv2d(
      in_channels = hdim,
      out_channels = vocab,
      kernel_size = 3
    ))
    decoder_layers.append(nn.ConvTranspose2d(
      in_channels = hdim,
      out_channels = 3,
      kernel_size = 3,
    ))
    
    self.encoder = nn.Sequential(*encoder_layers)
    self.codebook = nn.Embedding(vocab, embedding_dim)
    self.decoder = nn.Sequential(*decoder_layers)
      
  def forward(self, x):
    enc = self.encoder(x)
    soft_one_hot = F.gumbel_softmax(enc, tau = 1.)
    hid_tokens = torch.einsum("bnwh,nd->bdwh", soft_one_hot, self.codebook.weight)
    out = self.decoder(hid_tokens)
    return out


class EncoderBlock(nn.Module):
  def __init__(self, hidden_dim, out_channels, act = "relu"):
    super().__init__()
    self.resblock = nn.Sequential(
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
      nn.LeakyReLU(inplace=True),
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
    )
    self.down_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, stride=2)
    if act == "relu":
      self.act = nn.LeakyReLU(inplace = True)
    else:
      self.act = None

  def forward(self, x):
    out = x + self.resblock(x)
    out = self.down_conv(out)
    if self.act is not None:
      out = self.act(out)
    return out
  
class DecoderBlock(nn.Module):
  def __init__(self, hidden_dim, out_channels, act = "relu"):
    super().__init__()
    self.resblock = nn.Sequential(
      nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
      nn.LeakyReLU(inplace=True),
      nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
    )
    self.down_conv = nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2)
    if act == "relu":
      self.act = nn.LeakyReLU(inplace = True)
    elif act == "tanh":
      self.act = nn.Tanh()
    else:
      self.act = None

  def forward(self, x):
    out = x + self.resblock(x)
    out = self.down_conv(out)
    if self.act is not None:
      out = self.act(out)
    return out

class DiscreteResidualVAE(nn.Module):
  def __init__(self, hidden_dim, n_layers, num_embeds, in_channels = 3):
    super().__init__()
    encoder = []
    decoder = []
    for i in range(n_layers-1):
      encoder.append(EncoderBlock(
        hidden_dim = in_channels if i == 0 else hidden_dim,
        out_channels = hidden_dim
      ))
      decoder.append(DecoderBlock(
        hidden_dim=hidden_dim,
        out_channels=hidden_dim
      ))
    encoder.append(EncoderBlock(hidden_dim, num_embeds, None))
    decoder.append(DecoderBlock(hidden_dim, in_channels, "tanh"))
    
    self.encoder = nn.Sequential(*encoder)
    self.quantised = nn.Embedding(num_embeds, hidden_dim)
    self.decoder = nn.Sequential(*decoder)
    
  def forward(self, x):
    out = self.encoder(x)
    # the output from encoder has shape [bnhw] and so we must perform
    # softmax over n (number of embeddings/vocab size) and so dim = 1
    out = F.gumbel_softmax(out, tau = 1., dim = 1)
    out = einsum("bnhw,nd->bdhw", out, self.quantised.weight)
    out = self.decoder(out)
    loss = F.mse_loss(x, out)
    return None, loss, out

# ------- trainer 
class DiscreteVAETrainer:
  def __init__(self, model, train, test = None):
    self.model = model
    # self.train_dataset = Cifar(train=True, img_only=True)  # define the train dataset
    # self.test_dataset = Cifar(train=False, img_only=True)  # define the test dataset
    self.train_dataset = train
    self.test_dataset = test
    self.device = "cpu"
    if torch.cuda.is_available():
      self.device = torch.cuda.current_device()
      self.model = torch.nn.DataParallel(self.model).to(self.device)
      print("Model is now CUDA!")

  def save_checkpoint(self, ckpt_path = None):
    raw_model = self.model.module if hasattr(self.model, "module") else self.model
    ckpt_path = ckpt_path if ckpt_path is not None else self.config.ckpt_path
    print(f"Saving Model at {ckpt_path}")
    torch.save(raw_model.state_dict(), ckpt_path)

  def train(self, bs, n_epochs, lr, unk_id, save_every=500, test_every = 500):
    model = self.model
    train_data = self.train_dataset
    test_data = self.test_dataset
    epoch_step = len(train_data) // bs + int(len(train_data) % bs != 0)
    save_every = min([save_every, epoch_step])
    num_steps = epoch_step * n_epochs
    prev_loss = 10000
    no_improve_step = 0
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    gs = 0
    train_losses = [-1]
    model.train()
    for epoch in range(n_epochs):
      # ----- train for one complete epoch
      dl = DataLoader(
        dataset=train_data,
        batch_size=bs,
        pin_memory=True
      )
      pbar = trange(epoch_step)
      v = False
      for d, e in zip(dl, pbar):
        d = d.to(self.device)
        pbar.set_description(f"[TRAIN - {epoch}] GS: {gs}, Loss: {round(train_losses[-1], 5)}")
        _, loss, _ = model(d)
        loss = loss.mean() # gather from multiple GPUs
        if WANDB:
          wandb.log({"loss": loss.item()})

        # gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        gs += 1
        train_losses.append(loss.item())
        
        if gs and gs % save_every == 0:
          self.save_checkpoint(ckpt_path=f"models/vae_{unk_id}{gs}.pt")

        # ----- test condition
        if test_data != None and gs and gs % test_every == 0:
          dl = DataLoader(
            dataset=test_data,
            batch_size=bs,
            pin_memory=True
          )
          epoch_step_test = len(test_data) // bs + int(len(train_data) % bs != 0)
          pbar = trange(epoch_step_test)
          test_loss = []
          for d, e in zip(dl, pbar):
            d = d.to(self.device)
            pbar.set_description(f"[TEST - {epoch}]")
            _, loss, _ = model(d)
            loss = loss.mean() # gather from multiple GPUs
            test_loss.append(loss.item())

          test_loss = np.mean(test_loss)
          if WANDB:
            wandb.log({"test_loss": test_loss})

          print(":::: Loss:", test_loss)
          if prev_loss > test_loss:
            self.save_checkpoint(ckpt_path=f"models/vae_{unk_id}{gs}.pt")
            no_improve_step = 0
          else:
            no_improve_step += 1
          
          if no_improve_step == 3:
            print("::: No Improvements in 3 epochs, break training")
            break     
    
    print("EndSave")
    self.save_checkpoint(ckpt_path=f"models/vae_{unk_id}end.pt")
    with open("models/vae_loss.txt", "w") as f:
      f.write("\n".join([str(x) for x in train_losses]))


if __name__ == "__main__":
  args = argparse.ArgumentParser(description = "script to train the VectorQuantised-VAE")
  args.add_argument("--embedding_dim", type=int, default=64, help="embedding dimension to use")
  args.add_argument("--num_embeddings", type=int, default=512, help="number of embedding values to use")
  args.add_argument("--lr", type=float, default=0.0001, help="learning rate for the model")
  args.add_argument("--batch_size", type=int, default=300, help="minibatch size")
  args.add_argument("--n_epochs", type=int, default=30, help="minibatch size")
  args.add_argument("--model", type=str, default="res", choices=["res", "vqvae", "disvae"], help="minibatch size")
  args.add_argument("--dataset", type=str, default="flikr", choices=["flikr", "cifar"], help="minibatch size")
  args = args.parse_args()

  if args.model == "vqvae":
    print(":: Building VQVAE")
    model = VQVAE(
      in_channels = 3, 
      embedding_dim = args.embedding_dim,
      num_embeddings=args.num_embeddings,
      img_size = 32
    )
  elif args.model == "disvae":
    print(":: Building DiscreteVAE")
    model = DiscreteVAE(
      hdim=args.embedding_dim,
      num_layers=6,
      num_tokens=args.num_embeddings,
      embedding_dim=args.embedding_dim
    )
  elif args.model == "res":
    print(":: Building DiscreteResidualVAE")
    model = DiscreteResidualVAE(
      hidden_dim=args.embedding_dim,
      n_layers=3,
      num_embeds=args.num_embeddings,
    )
  else:
    raise ValueError("model should be one of `res`, `disvae`, `vqvae`")
    
  cifar = False
  if args.dataset == "flikr":
    train = FlikrDataset("../flickr30k_images/", "../ImageNet-Datasets-Downloader/data/")
    test = FlikrDataset("../flickr30k_images/", "../ImageNet-Datasets-Downloader/data/", train = False)
    print(":: Loaded FlikrDataset", len(train), len(test))
    if len(train) == 0:
        cifar = True
    
  if args.dataset == "cifar" or cifar:
    print(":: Loading cifar dataset")
    train = Cifar(True, True)
    test = Cifar(False, True)

  local_run = ""
  if WANDB:
    wandb.init(project = "vq-vae")
    wandb.watch(model) # watch the model metrics
    local_run = str(uuid4())[:8] + "_"
    print(":: Local Run ID:", local_run)
  set_seed(4)
  print(":: Number of params:", sum(p.numel() for p in model.parameters()))
  trainer = DiscreteVAETrainer(model, train, test)
  trainer.train(bs=args.batch_size, n_epochs=args.n_epochs, lr=args.lr, unk_id=local_run)
