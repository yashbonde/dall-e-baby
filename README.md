# dall-e-baby

<img src="assets/header.png">

OpenAI's [dall-e](https://openai.com/blog/dall-e/) is a kick ass model that takes in a natural language prompt and generates an images based on that. Now I cannot recreate the complete Dall-E so I make the baby version of it trained in CIFAR10-100 dataset. If Dall-E is picasso this is well... shit.

## Training

First step is to train a discrete VAE which can be done by:
```
python3 discrete_vae.py
```

It turns out training a VAE is not an easy task I trained using SGD but the training was taking too long and kept collapsing. Adam with gradient clipping works best. After training over a 90 models I found out that the best model was with `res:64, batch_size:128, num_embedding:1024, mid_res:16`. The models with larger mid size ie. where the low dimensional resolution is <4x the original resolution. Below is a sample from above configuration:

<img src="assets/128_64_1024.gif">

Where as what happens with config `res:128, batch_size:128, num_embedding:1024, mid_res:16`

<img src="assets/128_1024.gif">
