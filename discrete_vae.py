"""
code to train a discrete VAE
"""
import os
import torch
import random
import argparse
import numpy as np
from PIL import Image
from uuid import uuid4
from tqdm import trange
from torch import nn, einsum
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

try:
  from discrete_models import *
except ImportError:
  pass

WANDB=os.getenv("WANDB")
if WANDB:
  import wandb

# https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- helper functions
def set_seed(seed):
  if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_images_in_folder(folder, ext = [".jpg", ".png"]):
  # this method is faster than glob
  all_paths = []
  for root,_,files in os.walk(folder):
    for f in files:
      for e in ext:
        if f.endswith(e):
          all_paths.append(os.path.join(root,f))
  return all_paths

# --- dataset
class BabyDallEDataset(Dataset):
  def __init__(
      self,
      folders,
      train=True,
      train_split=0.994,
      res=102
    ):
    # since the test split is simply used for visualisation purposes, gives a
    # rough idea of where the training is headed it is fine if it is super
    # small

    # in the initial version of the model we used different arguments for each
    # folder now automatically load any arbitrary number of folder and images.
    # store metadata against each one of them for reference
    all_files = []
    meta = {}
    total = 0
    for name, path in folders.items():
      paths = get_images_in_folder(path)
      all_files.extend(paths)
      meta[name] = len(paths)
      total += len(paths)
    meta["total"] = total
    self.meta = meta

    # shuffle and segment dataset
    # during training we would like to get a larger variety of samples
    # so we include the RandomHorizontalFlip and RandomVerticalFlip
    np.random.shuffle(all_files)
    train_idx=int(train_split*len(all_files))
    if train:
      self.files=all_files[:train_idx]
      self.t=transforms.Compose([
          transforms.Resize((res, res)),
          transforms.CenterCrop(res),
          transforms.ToTensor(),
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.RandomVerticalFlip(p=0.5),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])
      self.mode = "train"
    else:
      self.files=all_files[train_idx:]
      self.t = transforms.Compose([
          transforms.Resize((res, res)),
          transforms.CenterCrop(res),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])
      self.mode = "test"
    self.meta[self.mode] = len(self.files)

  def __repr__(self):
    return f"<BabyDallEDataset ({self.mode}) " + "|".join([f"{k}:{v}" for k,v in self.meta.items()]) + ">"

  def __len__(self):
    return len(self.files)

  def __getitem__(self, i):
    """
    Does pytorch handle multiple workers automatically? Are duplicate samples
    returned. I don't think so. Test like this:
    ```
    class DSSimple(Dataset):
      def __init__(self):
        super().__init__()
        self.x = torch.arange(124)
      def __len__(self):
        return self.x.shape[0]
      def __getitem__(self, i):
        return self.x[i]
    
    for x in DataLoader(
        dataset = DSSimple(), batch_size = 10, num_workers = 4,
        shuffle = False, drop_last = True
      ):
      print(x)
    ```
    """

    # parallel = torch.utils.data.get_worker_info()
    # if parallel is None:
    #   # this is single worker process
    #   return self.t(Image.open(self.files[i]).convert('RGB'))
    
    # for parallel we use the sharding method where certain sections
    # are loaded by certain processes only
    # id, num_workers, seed, dataset = parallel

    return self.t(Image.open(self.files[i]).convert('RGB'))

# ----- model
class ResidualLayer(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ResidualLayer, self).__init__()
    self.resblock = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
    )

  def forward(self, input):
    return input + self.resblock(input)


class QuickGELU(nn.Module):
  # https://github.com/openai/CLIP/blob/main/clip/model.py
  def forward(self, x):
    return x * torch.sigmoid(1.702 * x)


class VQVAE_Encoder_v2(nn.Module):
  def __init__(
      self,
      in_channels,
      hidden_dims,
      num_embeddings,
      add_residual=True,
      **kwargs
  ):
    """
    The encoder structure is very straight forward, unlike the original implementation
    of VQVAE, we instead have a single block that looks like this:
    [
      [DownConv -> BN -> LReLU -> Residual -> LRelu] * hidden_dims]
      [Conv(hid, num_embeddings)]
    ]
    """
    super().__init__()
    modules = []
    for h_dim in hidden_dims:
      modules.append(nn.Sequential(
        nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(h_dim),
        QuickGELU()
      ))
      in_channels = h_dim
      if add_residual:
        modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(QuickGELU())

    modules.append(nn.Sequential(
        nn.Conv2d(in_channels, num_embeddings, kernel_size=1, stride=1),
        # QuickGELU()
    ))

    self.encoder = nn.Sequential(*modules)

  def forward(self, x):
    return self.encoder(x)


class VQVAE_Decoder_v2(nn.Module):
  def __init__(
      self,
      embedding_dim,
      hidden_dims,
      out_channels,
      add_residual=True,
      **kwargs
  ):
    """
    The decoder structure is very straight forward, unlike the original implementation
    of VQVAE, we instead have a single block that looks like this:
    [
      [Conv(embedding_dim, hid)]
      [Residual -> LReLU -> UpConv -> LRelu/Tanh] * hidden_dims]
    ]
    """
    super().__init__()
    modules = []
    modules.append(nn.Sequential(
      nn.ConvTranspose2d(embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(hidden_dims[-1]),
      QuickGELU()
    ))
    hidden_dims.reverse()
    hidden_dims = hidden_dims + [out_channels]  # add the last layer in it

    last = len(hidden_dims) - 1
    for i in range(last):
      if add_residual:
        modules.append(ResidualLayer(hidden_dims[i], hidden_dims[i]))
        modules.append(QuickGELU())

      # we need a tanh activation for the last layer not LReLU
      if i != last-1:
        modules.append(nn.Sequential(
          nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(hidden_dims[i + 1]),
          QuickGELU()
        ))
      else:
        modules.append(nn.Sequential(
          nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=4, stride=2, padding=1),
          nn.Tanh()
        ))

    self.decoder = nn.Sequential(*modules)

  def forward(self, x):
    return self.decoder(x)


class VQVAE_v3(nn.Module):
  def __init__(
      self,
      in_channels: int,
      embedding_dim: int,
      num_embeddings: int,
      hidden_dims=[128, 256],
      add_residual=True,
      **kwargs
    ):
    super().__init__()

    self.embedding_dim=embedding_dim
    self.num_embeddings=num_embeddings

    self.encoder = VQVAE_Encoder_v2(
      in_channels=in_channels,
      hidden_dims=hidden_dims,
      embedding_dim=embedding_dim,
      num_embeddings=num_embeddings,
      add_residual=add_residual,
    )
    self.codebook = nn.Embedding(num_embeddings, embedding_dim)
    self.decoder = VQVAE_Decoder_v2(
      embedding_dim=embedding_dim,
      hidden_dims=hidden_dims,
      out_channels=in_channels,
      add_residual=add_residual,
    )

  def _encode_image(self, input):
    encoding = self.encoder(input)
    print("!@#!@#", encoding.size())
    softmax = F.softmax(encoding, dim=1)
    print(softmax.size())
    softmax = F.one_hot(
        torch.argmax(
          softmax, dim=1),
        num_classes=softmax.size(1)
    ).permute((0, 3, 1, 2)).float()
    encoding_ids = torch.argmax(softmax, dim=1).view(encoding.size(0), -1)
    return encoding_ids

  def _decode_ids(self, softmax = None, image_tokens = None):
    if softmax is not None:
      quantized_inputs = einsum("bdhw,dn->bnhw", softmax, self.codebook.weight)
    elif image_tokens is not None:
      res = np.sqrt(image_tokens.shape[1]).astype(int)
      bs = image_tokens.size(0)
      quantized_inputs = self.codebook(image_tokens).view(bs, res, res, -1)
      quantized_inputs = quantized_inputs.permute((0, 3, 1, 2))
    recons = self.decoder(quantized_inputs)
    return recons

  def norm_img(self, img):
    img -= img.min()
    img /= img.max()
    return img

  def forward(self, input, v=False):
    # first step is to pass the image through encoder and get the embeddings
    encoding=self.encoder(input)

    # the idea behind this model is that encoder is supposed to generate a dense
    # embedding (smaller image) with larger vocabulary and more information
    # (embedding dimension). The decoder is then supposed to take in this
    # image and then recreate the image, this is generally not possible with
    # standard argmax method and so we need to use the reparameterization trick
    # from: https://arxiv.org/pdf/1611.01144.pdf
    # which makes argmax operation differentiable. However it does to at the added
    # cost of a noise that is introduced in the model. This added noise in any
    # other case might have been an issue but with VAEs it suits perfectly.
    if self.training:
      softmax = F.gumbel_softmax(encoding, tau = 1., hard = True, dim = 1)
    else:
      # during testing we can ignore this added noise and go for standard softmax
      # now this does not give use the exact sharp values that hard-gumbel gives but
      # is still usable.
      softmax = F.softmax(encoding, dim = 1)

      # we can also improve this by introducing the argmax method, since we know that
      # during testing we do not need the gradient we can get away with it. We simply
      # create the same shape as we would get in the Gumbel distribution

      # The previous code I used for scatter was wrong, meaning in all the models
      # the test image generated was far worse than what the model was actually
      # producing
      softmax = F.one_hot(
        torch.argmax(softmax, dim = 1),
        num_classes = softmax.size(1)
      ).permute((0, 3, 1, 2)).float()

    # we can also convert the softmax distribution to argmax and identify the final ids
    # to use for language model
    encoding_ids = torch.argmax(softmax, dim = 1).view(encoding.size(0), -1)

    # now just like conventional embedding we tell the model to fill the hard indices
    # with the values from a learnable embedding matrix.
    quantized_inputs = einsum("bdhw,dn->bnhw", softmax, self.codebook.weight)

    # vq loss or Vector Quantisation loss as given in the original VQ-VAE paper
    # from: https://arxiv.org/pdf/1711.00937.pdf
    # we do not need this in out case since we use gumbel as a bypass for it.
    # vq_loss = F.mse_loss(quantized_inputs.detach(), encoding) + \
    #   0.25 * F.mse_loss(quantized_inputs, encoding.detach())

    # we pass the embedding through decoder that is supposed to recreate the image
    recons=self.decoder(quantized_inputs)
    recons_loss=F.mse_loss(recons, input)
    loss = recons_loss # + vq_loss
    return encoding_ids, loss, recons


# ------- trainer
class DiscreteVAETrainer:
  def __init__(self, model, train, test=None):
    self.model=model
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

  def train(
      self, bs, n_epochs, lr, folder_path, skip_steps,
      test_every=500, test_batch_size=None, save_every=None,
      gradient_accumulation_steps = 1.,
      **kwargs
    ):
    model=self.model
    train_data=self.train_dataset
    test_data=self.test_dataset
    epoch_step=len(train_data) // bs + int(len(train_data) % bs != 0)
    optim=torch.optim.Adam(model.parameters(), lr=lr)
    gs=0
    train_losses=[-1]
    model.train()
    do_skip = True
    for epoch in range(n_epochs):
      # ----- train for one complete epoch
      dl=DataLoader(
        dataset=train_data,
        batch_size=bs,
        pin_memory=True,    # for CUDA
        shuffle=True,       # of course, my stupid ass didn't do it for first 74 runs
        num_workers=1       # number of workers for parallel loading 
      )
      pbar=trange(epoch_step)
      for d, loop_idx in zip(dl, pbar):
        # don't train if we need to skip some steps but we do not want
        # it to skip for all the future epochs and so we add `do_skip` flag
        if skip_steps and loop_idx < skip_steps and do_skip:
          continue
        do_skip = False

        # train the model
        d=d.to(self.device)
        pbar.set_description(f"[TRAIN - {epoch}] GS: {gs}, Loss: {round(train_losses[-1], 5)}")
        _, loss, _=model(d)
        loss=loss.mean() # gather from multiple GPUs
        if WANDB:
          wandb.log({"loss": loss.item()})

        # gradient clipping
        loss = loss / gradient_accumulation_steps
        loss.backward()
        if gs and gs % gradient_accumulation_steps == 0:
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
          optim.step()
        gs += 1
        train_losses.append(loss.item())

        # ----- test condition
        if test_data != None and gs and gs % test_every == 0:
          print(":: Entering Testing Mode")
          if test_batch_size is None:
            test_batch_size = bs * 4
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
          plt.savefig(f"{folder_path}/sample_{gs}.png")
          del fig # delete and save the warning

          test_loss=np.mean(test_loss)
          if WANDB:
            wandb.log({"test_loss": test_loss})
          print(":::: Loss:", test_loss)

        if gs and save_every and gs % save_every == 0:
          self.save_checkpoint(ckpt_path=f"{folder_path}/vae_{gs}.pt")
        # ------ testing ends
        model.train()  # convert model back to training mode

    print("EndSave")
    self.save_checkpoint(ckpt_path=f"{folder_path}/vae_end.pt")


# def init_weights(module):
#   if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.ConvTranspose2d)):
#     module.weight.data.normal_(mean=0.0, std=0.02)
#     if isinstance(module, nn.Linear) and module.bias is not None:
#       module.bias.data.zero_()
#   elif isinstance(module, (nn.LayerNorm)):
#     module.bias.data.zero_()
#     module.weight.data.fill_(1.0)


def init_weights(module):
  """Initialise all the weights by standard deviation = root of channel"""
  if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
    weight_dim = module.weight.size(1)
    nn.init.normal_(module.weight, std=weight_dim ** -0.5)
    if module.bias is not None:
      module.bias.data.zero_()
  elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
    module.bias.data.zero_()
    module.weight.data.fill_(1.0)


if __name__ == "__main__":
  args=argparse.ArgumentParser(description="script to train a Discrete VAE model")
  args.add_argument("--embedding_dim", type=int, default=300, help="embedding dimension to use")
  args.add_argument("--res", type=int, default=128, help="resolution of the image")
  args.add_argument("--num_embeddings", type=int, default=3000, help="number of embedding values to use")
  args.add_argument("--n_layers", type=int, default=4, help="number of layers in the model")
  args.add_argument("--lr", type=float, default=2e-4, help="learning rate for the model")
  args.add_argument("--test_every", type=int, default=900, help="test model after these steps")
  args.add_argument("--save_every", type=int, default=1800, help="save model after these steps")
  args.add_argument("--batch_size", type=int, default=64, help="minibatch size")
  args.add_argument("--n_epochs", type=int, default=2, help="number of epochs to train for")
  args.add_argument(
    "--model", type=str, default="vqvae3",
    choices=["res", "vqvae", "disvae", "vqvae2", "vqvae3"],
    help="model architecture to use"
  )
  args.add_argument("--add_residual", type=bool, default=False, help="to use the residual connections")
  args.add_argument(
    "--dataset", type=str, default="mix",
    choices=["mix", "cifar"],
    help="dataset version to use"
  )
  args.add_argument("--skip_steps", type=int, default=0, help="number of steps to skip when restarting")
  args.add_argument("--restart_path", type=str, default=None, help="path to the model that is to be restarted")
  args.add_argument("--seed", type=int, default=3, help="seed value") # 3 = my misha
  args.add_argument(
    "--gradient_accumulation_steps", type=int, default=2.,
    help="perform backward pass after these global steps"
  )
  args=args.parse_args()

  # set seed to ensure everything is properly split
  set_seed(args.seed)
  folder_path = f"./{args.model}_{args.res}_{args.embedding_dim}_"+\
    f"{args.num_embeddings}_{int(args.add_residual)}_{args.batch_size}"
  print(f":: Will Save data in {folder_path}")
  os.makedirs(folder_path, exist_ok=True)

  if args.skip_steps != 0 and args.restart_path == None:
    raise ValueError("Need to provide --restart_path when restarting")

  if args.model == "vqvae":
    print(":: Building VQVAE")
    model=VQVAE(
      in_channels=3,
      embedding_dim=args.embedding_dim*4,
      num_embeddings=args.num_embeddings,
      img_size=args.res,
      hidden_dims=[args.embedding_dim, args.embedding_dim, args.embedding_dim * 2],
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
    print(":: Building VQVAEv2")
    model=VQVAEv2(
      hid=args.embedding_dim,
      vocab_size=args.num_embeddings
    )
  elif args.model == "vqvae3":
    print(":: Building VQVAE_v3")
    if args.res == 64:
        hdim = [args.embedding_dim, args.embedding_dim * 2]
    elif args.res == 128:
        hdim = [args.embedding_dim, int(args.embedding_dim * 1.5), args.embedding_dim * 2]
    elif args.res == 256:
        hdim = [args.embedding_dim, int(args.embedding_dim * 1.5), args.embedding_dim * 2, int(args.embedding_dim * 2.5)]
    model=VQVAE_v3(
      in_channels=3,
      embedding_dim=args.embedding_dim*3,
      num_embeddings=args.num_embeddings,
      img_size=args.res,
      hidden_dims=hdim,
      add_residual=args.add_residual
    )
  else:
    raise ValueError("incorrect model run $python3 discrete_vae.py --help")
  
  # print(model)
  print(":: Number of params:", sum(p.numel() for p in model.parameters()))
  resume = False
  if args.restart_path is not None:
    model.load_state_dict(torch.load(args.restart_path))
    resume = True
    print(f":: Found restart_path {args.restart_path} | Setting resume: {resume}")

  # let's not initialise the weights and pytorch take care of it
  model.apply(init_weights) # initialise weights

  # now that we can load any number of datasets from different folder
  # this is the master copy of all the folders
  folders = {
    "openimages256": "../downsampled-open-images-v4/",
    "food-101": "../food-101/",
    "svhn": "../housenumbers/",
    "indoor": "../indoorCVPR/",
    "imagenet_train64x64": "../small/",
    "stl10": "../stl10/",
    "genome1": "../VG_100K/",
    "genome2": "../VG_100K_2/",
  }

  if args.dataset == "mix":
    # use the mixture dataset
    print(":: Loading Mixture BabyDallEDataset")
    train=BabyDallEDataset(
      folders=folders,
      res=args.res,
      train=True
    )
    test=BabyDallEDataset(
      folders=folders,
      res=args.res,
      train=False
    )
    print(":: Dataset:", train) # prints the metrics for this dataset
    if len(train) == 0:
        print(":: [!] Found no data in Mixture BabyDallEDataset please ensure you have the data!")
        exit()

  elif args.dataset == "cifar":
    # use cifar dataset
    print(":: Loading cifar dataset")
    train=Cifar(True, True)
    test=Cifar(False, True)

  local_run=""
  if WANDB:
    wandb.init(project="vq-vae", resume = resume)
    wandb.watch(model) # watch the model metrics
    local_run=str(uuid4())[:8] + "_"
    print(":: Local Run ID:", local_run)
  trainer=DiscreteVAETrainer(model, train, test)
  trainer.train(
    bs=args.batch_size,
    n_epochs=args.n_epochs,
    lr=args.lr,
    skip_steps=args.skip_steps,
    unk_id=local_run,
    test_every=args.test_every,
    save_every=args.save_every,
    folder_path=folder_path
  )
