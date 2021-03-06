# script to generate images from the given sentence

import os
import json
import torch
import hashlib
import numpy as np
from PIL import Image
from argparse import ArgumentParser

from dalle import DallETransformer, Dalle, Vqvae, set_seed, Tokenizer


def md5(x): return hashlib.md5(x.encode("utf-8")).hexdigest()

def toImage(x):
  return Image.fromarray((x.numpy() * 255).astype(np.uint8))


if __name__ == "__main__":
  args = ArgumentParser(description="Generate image using dall-e-baby")
  args.add_argument("--dalle-path", type=str, help="path to dalle weights")
  args.add_argument("--vqvae-path", type=str, help="path to VQVAE weights")
  args.add_argument("--tokenizer", type=str, help="path to tokenizer")
  args.add_argument("--seed", type=int, default=4,
                    help="Seed value for generation")
  args.add_argument("--folder", type=str, default="./generated",
                    help="folder to store generated images in")
  args = args.parse_args()

  folder_path = os.path.expanduser(args.folder)
  os.makedirs(folder_path, exist_ok=True)

  meta_cache_path = os.path.join(folder_path, "generated_meta.json")
  mode = "w" if not os.path.exists(meta_cache_path) else "r"
  with open(meta_cache_path, mode) as f:
    try:
      if mode == "r":
        meta_cache = json.load(f)
    except:
      meta_cache = {}

  vqvae = Vqvae(args.vqvae_path)
  set_seed(args.seed)

  # load the tokenizer
  tok = Tokenizer.from_file(args.tokenizer)
  text_end_id = tok.get_vocab()["<|endoftext|>"]
  image_end_id = tok.get_vocab()["<|endofimage|>"]
  print("Tokenizer loaded with vocab size:", tok.get_vocab_size())

  # define config and load the model
  dalle_args = Dalle.parse_name(
      model_path=args.dalle_path,
      image_vocab_size=vqvae.num_embeddings,
      text_vocab_size=tok.get_vocab_size(),
  )
  dalle = DallETransformer(vqvae.get_model(), dalle_args)
  map_location = "cpu" if not torch.cuda.is_available() else "cuda"
  print(f": Loading model to {map_location}")
  dalle.load_state_dict(torch.load(
      args.dalle_path,
      map_location=map_location
  ))
  dalle = dalle.to(map_location)
  dalle.eval()

  print(f"Loaded the model, entering loop. Saving images in {folder_path}")
  while True:
    # continue in the loop so user can keep using this
    input_prompt = input("('q' to exit) >>> ")
    if input_prompt.lower() in ["q"]:
      with open(meta_cache_path, "w") as f:
        f.write(json.dumps(meta_cache))
      break

    cap = input_prompt.lower() * 100
    text_tokens = tok.encode(
        cap).ids[:dalle_args.text_context_len - 1] + [text_end_id]
    text_tokens = torch.Tensor(text_tokens).unsqueeze(
        0).long().to(map_location)

    print("Starting Generation (Scratch)", "-"*70)
    output_images, scores = dalle.complete_image(
        text_tokens=text_tokens,
        num_return_sequences=2,
        top_k=100,
        top_p=0.95,
        temperature=0.99,
        _verbose=False
    )

    # define the unique hash, iterate in case such a prompt already exists
    _hash = md5(input_prompt)
    if _hash in meta_cache:
      incr = 0
      while _hash in meta_cache:
        _hash = md5(input_prompt + str(incr))
        incr += 1

    # save the images
    this_image_paths = []
    for idx, (o, s) in enumerate(zip(output_images, scores)):
      path = os.path.join(folder_path, f"{_hash}_{idx}_{s:.4f}.jpg")
      img = toImage(o.cpu())
      img.save(path)
      this_image_paths.append(path)
    meta_cache[_hash] = {
        "paths": this_image_paths,
        "prompt": input_prompt
    }  # update cache for storing information
