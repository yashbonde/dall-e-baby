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
    if i > len(self.c100):
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
        
  def forward(self, x):
    enc = self.encoder(x)
    soft_one_hot = F.gumbel_softmax(enc, tau = 1.)
    hid_tokens = einsum("bnwh,nd->bdwh", soft_one_hot, self.codebook.weight)
    out = self.decoder(hid_tokens)
    loss = F.mse_loss(out, x)
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
    optim = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.99, weight_decay = 1e-5)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [1000, 100000])

    gs = 0
    model.train()
    for e in range(n_epochs):
      dl = DataLoader(
        dataset=train_data,
        batch_size=bs,
        pin_memory=True
      )
      pbar = trange(epoch_step)
      train_losses = [-1]
      for d, e in zip(dl, pbar):
        # d = trans(d)
        d = d.to(self.device)
        pbar.set_description(f"[TRAIN] GS: {gs}, Loss: {round(train_losses[-1], 5)}")
        _, loss = model(d)
        loss = loss.mean() # gather from multiple GPUs

        loss.backward()
        optim.step()
        lr_scheduler.step()

        train_losses.append(loss.item())

        if gs and gs % save_every == 0:
          self.save_checkpoint(ckpt_path = "models/vae.pt")


if __name__ == "__main__":
  model = DiscreteVAE(
    hdim=128,
    num_layers=6,
    num_tokens=1024,
    embedding_dim=256
  )
  print(model)

  trainer = DiscreteVAETrainer(model)
  trainer.train(32, 1)
