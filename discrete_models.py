"""
these are the deprecated models after extensive testing.
"""
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset


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
      embedding_dim=embedding_dim,
      num_embeddings=num_embeddings
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


# ---- dataset
CIFAR_MEAN = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
CIFAR_STDS = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]


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
