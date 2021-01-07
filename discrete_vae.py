"""
code to train a discrete VAE
"""
import torch
from tqdm import trange
from einops import rearrange
from torch import nn, einsum
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from vq_vae import VQVAE

class CifarTrain(Dataset):
  def __init__(self, img_only = False):
    self.c100 = datasets.CIFAR100("./data", download = True)
    self.c10 = datasets.CIFAR10("./data", download = True)
    self.img_only = img_only
    self.transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
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


class DiscreteVAE(nn.Module):
  def __init__(self, hdim = 128, num_layers = 6, num_tokens = 1024, embedding_dim = 256):
    super().__init__()
    encoder_layers = []
    decoder_layers = []
    for i in range(num_layers):
      enc_layer = nn.Conv2d(
        in_channels = hdim if i > 0 else 3,
        out_channels = hdim,
        kernel_size = 4,
        stride = 1,
        padding = 0,
        dilation = 1,
      )
      encoder_layers.extend([enc_layer, nn.ReLU()])
      
      dec_layer = nn.ConvTranspose2d(
        in_channels = embedding_dim if i == 0 else hdim,
        out_channels = hdim,
        kernel_size = 4,
        stride = 1,
        padding = 0,
        dilation = 1,
      )
      decoder_layers.extend([dec_layer, nn.ReLU()])
        
    encoder_layers.append(nn.Conv2d(
      in_channels = hdim,
      out_channels = num_tokens,
      kernel_size = 3
    ))
    decoder_layers.append(nn.ConvTranspose2d(
      in_channels = hdim,
      out_channels = 3,
      kernel_size = 3,
    ))
    
    self.encoder = nn.Sequential(*encoder_layers)
    self.codebook = nn.Embedding(num_tokens, embedding_dim)
    self.decoder = nn.Sequential(*decoder_layers)
        
  def forward(self, x, v = False):
    enc = self.encoder(x)
    if v: print(enc)
    soft_one_hot = F.gumbel_softmax(enc, tau = 1.)
    if v: print(soft_one_hot)
    hid_tokens = einsum("bnwh,nd->bdwh", soft_one_hot, self.codebook.weight)
    if v: print(hid_tokens)
    out = self.decoder(hid_tokens)
    if v: print(out)
    loss = F.mse_loss(out, x)
    if v: print(loss)
    return soft_one_hot, loss


class DiscreteVAETrainer:
  def __init__(self, model):
    self.model = model
    self.train_dataset = CifarTrain(True)  # define the dataset
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

  def train(self, bs, n_epochs, save_every=1000):
    model = self.model
    train_data = self.train_dataset
    epoch_step = len(train_data) // bs + int(len(train_data) % bs != 0)
    num_steps = epoch_step * n_epochs
    optim = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 0.1)

    gs = 0
    train_losses = [-1]
    model.train()
    for e in range(n_epochs):
      dl = DataLoader(
        dataset=train_data,
        batch_size=bs,
        pin_memory=True
      )
      pbar = trange(epoch_step)
      v = False
      for d, e in zip(dl, pbar):
        d = d.to(self.device)
        pbar.set_description(f"[TRAIN] GS: {gs}, Loss: {round(train_losses[-1], 5)}")
        # _, loss = model(d, v = v)
        loss = model.loss_function(
          *model(d)
        )
        loss = loss.mean() # gather from multiple GPUs
        if v:
          exit()
        if torch.isnan(loss).any():
          print()
          v = True

        loss.backward()
        optim.step()
        gs += 1

        train_losses.append(loss.item())

        if gs and gs % save_every == 0:
          self.save_checkpoint(ckpt_path = f"models/vae_{gs}.pt")
    
    print("EndSave")
    self.save_checkpoint(ckpt_path = "models/vae_end.pt")
    with open("models/vae_loss.txt", "w") as f:
      f.write("\n".join([str(x) for x in train_losses]))


if __name__ == "__main__":
  # model = DiscreteVAE(
  #   hdim=64,
  #   num_layers=6,
  #   num_tokens=1028,
  #   embedding_dim=128
  # )
  model = VQVAE(
    in_channels = 3, 
    embedding_dim = 128,
    num_embeddings = 512,
    hidden_dims = 128
  )
  print(":: Number of params:", sum(p.numel() for p in model.parameters()))
  trainer = DiscreteVAETrainer(model)
  trainer.train(32, 3)
