# script to test generation methods

import torch
from types import SimpleNamespace
from dalle import DallETransformer, TransformerConfig, init_weights
from discrete_vae import VQVAE_v3, set_seed

set_seed(1)

test_args = SimpleNamespace(
  embedding_dim = 8,          # VQVAE embeding dim
  num_embeddings = 128,       # number of tokens in codebook for VQVAE
  res = 32,                   # resolution of input 
  add_residual = False,       # no residual connections in VQVAE
  text_context_len = 16,      # number of text tokens
  n_embd = 16,                # embedding dimension for transformer
  n_layers = 2,               # number of layers in transformer
  n_heads = 2,                # number of heads in MHA
  text_vocab_size = 128,      # vocabulary size of text
  hidden_dims = [8, 8, 8],    # hidden dimensions in VQVAE
  total_context_len = 16 + 16 # total context len
)

print(test_args.total_context_len)

# define the VQVAE model 
model_vae = VQVAE_v3(
    in_channels=3,
    embedding_dim=test_args.embedding_dim*3,
    num_embeddings=test_args.num_embeddings,
    img_size=test_args.res,
    hidden_dims=test_args.hidden_dims,
    add_residual=test_args.add_residual
)

# transformer configuration
transformer_config = TransformerConfig(
    text_context_len=test_args.text_context_len,
    image_context_len=test_args.total_context_len - test_args.text_context_len,
    text_vocab_size=test_args.text_vocab_size,
    image_vocab_size=test_args.num_embeddings,
    n_embd=test_args.n_embd,
    n_layers=test_args.n_layers,
    n_heads=test_args.n_heads
)

# transformer model
dalle = DallETransformer(
  model_vae,
  transformer_config
)
init_weights(dalle)

pytorch_total_params = sum(p.numel() for p in dalle.parameters())
print("---<>", pytorch_total_params)

batch_size = 1
d = {
  "text_tokens": torch.randint(low=0, high=test_args.text_vocab_size, size=(batch_size, test_args.text_context_len))
}
print({k:v.size() for k,v in d.items()})

# perform one run on the model
out = dalle(**d, loss = False)
print(out[0].size())

# generator can be called right from dalle
print("Starting Generation (Scratch)", "-"*70)
output_images, scores = dalle.complete_image(
  text_tokens=d["text_tokens"],
  num_return_sequences=3,
  top_k = 5,
  top_p = 0.95,
  temperature=0.95,
  _verbose = True
)
assert output_images.size(0) == 3 * batch_size
for o,s in zip(output_images, scores):
  print(s, o.size())

# we can also provied partial image and ask model to generate the image
print("Starting Generation (Prior Image)", "-"*70)
images =  torch.randn(batch_size, 3, test_args.res, test_args.res)
encoded = model_vae._encode_image(images)
print("Encoded Image:", encoded, encoded.size())

encoded = encoded[:, :encoded.size(1) // 2]
print("Image prior:", encoded, encoded.size())

output_images, scores = dalle.complete_image(
  text_tokens=d["text_tokens"],
  image_tokens = encoded,
  num_return_sequences=3,
  top_k = 5,
  top_p = 0.95,
  temperature=0.95,
  _verbose = True
)
assert output_images.size(0) == 3 * batch_size
for o,s in zip(output_images, scores):
  print(s, o.size())
