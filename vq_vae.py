"""
implemented version of: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
"""
import torch
from torch import nn, Tensor
from torch.nn import functional as F

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

    def __init__(
      self,
      in_channels: int,
      out_channels: int
    ):
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
              img_size: int = 64,
              **kwargs) -> None:
    super(VQVAE, self).__init__()

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

  def forward(self, input: Tensor, v = False):
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
