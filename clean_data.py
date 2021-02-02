import os
import subprocess
import numpy as np
from PIL import Image
from tqdm import trange
from multiprocessing import Process, cpu_count

# STL-10 files come as a binary that need to be converted to images to make them useful
print(":: Converting STL to Images")
stl10_binary_files = ['../stl10/stl10_binary/test_X.bin', '../stl10/stl10_binary/train_X.bin']
for f in stl10_binary_files:
  with open(f, 'rb') as fobj:
    # read whole file in uint8 chunks
    everything = np.fromfile(fobj, dtype=np.uint8)

    # We force the data into 3x96x96 chunks, since the
    # images are stored in "column-major order", meaning
    # that "the first 96*96 values are the red channel,
    # the next 96*96 are green, and the last are blue."
    # The -1 is since the size of the pictures depends
    # on the input file, and this way numpy determines
    # the size on its own.

    images = np.reshape(everything, (-1, 3, 96, 96))

    # Now transpose the images into a standard image format
    # readable by, for example, matplotlib.imshow
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    images = np.transpose(images, (0, 3, 2, 1))
  
  for idx, img in enumerate(images):
    img = Image.fromarray(img)
    fpath = f.replace(".bin", f"_{idx}.jpg")
    img.save(fpath)
  
  print(": Completed", f)

# sometimes you will see that the files are corrupted and so they need to be found and cleaned
# replace this with the datasets that you are going to use
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


def get_images_in_folder(folder, ext=[".jpg", ".png"]):
  # this method is faster than glob
  all_paths = []
  for root, _, files in os.walk(folder):
    for f in files:
      for e in ext:
        if f.endswith(e):
          all_paths.append(os.path.join(root, f))
  return all_paths


all_files = []
meta = {}
total = 0
for name, path in folders.items():
    paths = get_images_in_folder(path)
    all_files.extend(paths)
    meta[name] = len(paths)
    total += len(paths)
meta["total"] = total

print("-"*70)
for k,v in meta.items():
  print(k,"::",v)

def check_files(files):
  fails = []
  pbar = range(len(files))
  for i in pbar:
    f = files[i]
    try:
      Image.open(f)
    except:
      fails.append(f)
  if len(fails):
    print("\n\n", " ".join(fails))
  subprocess.run(["rm", *fails])

# split all the files into buckets and an extra bucket with the files not checked
print("-"*70)
print(":: Starting corruption check")
workers = cpu_count()
splits = np.split(
  np.array(all_files[:-(len(all_files) % workers)]),
  workers
) + [all_files[-(len(all_files) % workers):]]
print(":: Bucket sizes:", [len(x) for x in splits])

# now run all the checks in parallel
ps = []
for s in splits[:-1]:
  print(len(s))
  ps.append(Process(target=check_files, args=(s,)))
  ps[-1].start()

for p in ps:
  p.join()

# extra check for the small last bucket
check_files(splits[-1])
print(":: Process completed. Rechecking!")

all_files = []
meta = {}
total = 0
for name, path in folders.items():
    paths = get_images_in_folder(path)
    all_files.extend(paths)
    meta[name] = len(paths)
    total += len(paths)
meta["total"] = total

print("-"*70)
for k, v in meta.items():
  print(k, "::", v)
print("-"*70)
