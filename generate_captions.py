# Script to prepare the captions.json object that will be used for training transformer

import os
import re
import h5py
import json
import pandas as pd
from discrete_vae import *
from tabulate import tabulate


# ---- Datasets where captions are already given

# coco captions
def get_coco_captions(captions_path):
  with open(captions_path, "r") as f:
    cap2017 = json.load(f)
  captions = {}
  dropped = []
  for x in cap2017["annotations"]:
    id_ = str(x["image_id"])
    id_ = "0"*(12-len(id_))+id_
    path = "../coco/train2017/"+id_+".jpg"
    if not os.path.exists(path):
      dropped.append(path)
      continue

    key = "coco_"+str(x["image_id"])
    captions.setdefault(key, {
      "caption": "",
      "path": path
    })
    captions[key]["caption"] += " " + x["caption"]
  return captions, dropped


# visual Genome captions
def get_genome_captions(root_folder = "../VG_100K_2"):
  with open(f"{root_folder}/region_descriptions.json", "r") as f:
    regdes = json.load(f)
    
  captions = {}
  dropped = []
  for item in regdes:
    id = item["id"]
    path = f"../VG_100K_2/VG_100K/{id}.jpg"
    if not os.path.exists(path):
      path = f"../VG_100K_2/VG_100K_2/{id}.jpg"
    if not os.path.exists(path):
      dropped.append(id)
      continue
    captions["genome_"+str(item["id"])] = {
      "caption":" ".join([x["phrase"] for x in item["regions"]]),
      "path": path
    }
    
  return captions, dropped

# Flickr30k captions
def get_flickr30k_captions(rf="../flickr30k_images"):
  data = pd.read_csv(f"{rf}/flickr30k_images/results.csv", sep="|")
  captions = {}
  dropped = []
  for idx, (img_id, df_sub) in enumerate(data.groupby("image_name")):
    path = f"../flickr30k_images/flickr30k_images/{img_id}"
    if not os.path.exists(path):
      dropped.append(path)
      continue
    captions[f"flickr_{idx}"] = {
      "caption": " ".join([str(x) for x in df_sub[" comment"].values.tolist()]),
      "path": path
    }
  return captions, dropped


# ---- Datasets where only labels are given so we have to generate captions for this

def get_open_images_label_names():
  with open("../downsampled-open-images-v4/class-descriptions-boxable.csv", "r") as f:
    open_image_labels = {x.split(",")[0]: x.split(",")[1] for x in f.read().split("\n") if len(x)}
  return open_image_labels

def get_open_images_labels(annotations_path):
  open_image_labels = get_open_images_label_names()
  print("-->", annotations_path)
  df = pd.read_csv(annotations_path)
  image_to_labels = {}
  dropped = []
  pbar = trange(len(df.ImageID.unique()))
  path_f = "../downsampled-open-images-v4/256px/"
  if "validation" in annotations_path:
    path_f += "validation/"
  elif "train" in annotations_path:
    path_f += "train-256/"
  for _, (img_id, df_sub) in zip(pbar, df.groupby("ImageID")):
    sub_labels = df_sub[df_sub.Confidence == 1].LabelName.values.tolist()
    if not sub_labels:
      dropped.append(img_id)
    image_to_labels["open_images_" + img_id] = {
      "label": [open_image_labels[x] for x in sub_labels],
      "path": f"{path_f}{img_id}.jpg"
    }

  return image_to_labels, dropped


def get_indoor_cvpr(rf= "../indoorCVPR/"):
  indoor = get_images_in_folder(rf)
  img2label = {idx: {
    "label": [x.split("/")[2].replace("_", " ").title()],
    "path": x
  } for idx, x in enumerate(indoor)}
  return img2label


def get_food(rf="../food-101/"):
  food = get_images_in_folder(rf)
  img2label = {idx: {
    "label": [x.split("/")[3].replace("_", " ").title()],
    "path": x
  } for idx, x in enumerate(food)}
  return img2label


def get_stl10(bin_file = "../stl10/stl10_binary/train_y.bin"):
  classes = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
  with open(bin_file, 'rb') as fobj:
    # read whole file in uint8 chunks
    everything = np.fromfile(fobj, dtype=np.uint8)
  labels = [[classes[x - 1]] for x in everything]

  # sort the images in the STL10 that are already parsed
  stl10 = [x for x in get_images_in_folder("../stl10/stl10_binary/") if "train" in x]
  imgs = {int(x.split("_")[-1].split(".")[0]): x for x in stl10}
  img2label = {
    f"stl_{k}":{
      "path": imgs[k],
      "label": l
    } for k,l in zip(imgs, labels)
  }
  
  return img2label


def get_svhn_data(matfile = '../housenumbers/train/digitStruct.mat'):
  def readInt(intArray, dsFile):
    intRef = intArray[0]
    isReference = isinstance(intRef, h5py.Reference)
    intVal = 0
    if isReference:
      intObj = dsFile[intRef]
      intVal = int(intObj[0])
    else: # Assuming value type
      intVal = int(intRef)
    return intVal
  
  digitmat = h5py.File(matfile, 'r')
  
  print("Loading labels:")
  digit_struct = digitmat["digitStruct"]
  labels_to_return = []
  for _, box in zip(trange(digit_struct["bbox"].shape[0]), digit_struct["bbox"]):
    bbox = digitmat[box[0]] # load bbox using reference
    labels = bbox["label"] # bbox object has the data for bounding box and labels
    lbl = "".join([
      str(readInt(l, digitmat))
      for l in labels
    ])           # create the label string by iterating over all the bboxes
    labels_to_return.append(lbl)

  # now replicate for names
  print("Loading filenames:")
  names = []
  for _, name in zip(trange(digit_struct["name"].shape[0]), digit_struct["name"]):
    name = ''.join(chr(i) for i in digitmat[name[0]])
    names.append(name)
    
  # create final mapping
  dropped =[]
  img2label = {}
  for name, label in zip(names, labels_to_return):
    path = f"../housenumbers/train/{name}"
    if not os.path.exists(path):
      dropped.append(path)
      continue
    img2label[f"housenumber_{name.split('.')[0]}"] = {
      "path": path,
      "label": [label]
    }
    
  return img2label, dropped

# ---- Captions are generated using CaptionsGenerator

class CaptionGenerator():
  templates_labels = [
    "a picture of {}",
    "a photo that has {}",
    "photo consisting of {}",
    "a low resolution photo of {}",
    "small photo of {}",
    "high resolution picture of {}",
    "low resolution picture of {}",
    "high res photo that has {}",
    "low res photo of {}",
    "{} in a photo",
    "{} in a picture",
    "rendered picture of {}",
    "jpeg photo of {}",
    "a cool photo of {}",
    "{} rendered in a picture",
  ]

  templates_indoor = [
    "indoor picture of {}",
    "picture inside of {}",
    "picture of {} from inside",
  ]

  templates_food = [
    "picture of {}, a food item",
    "photo of food {}",
    "nice photo of food {}",
    "picture of food item {}",
    "picture of dish {}",
    "picture of {}, a food dish",
    "gourmet food {}",
  ]

  templates_svhn = [
    "a picture of house number '{}'",
    "number '{}' written in front of a house",
    "street house number '{}' written on a door",
    "a photo with number '{}' written in it",
    "number '{}' written on a door",
    "photograph of number '{}'"
  ]

  captions_templates = {
    "open_images": [templates_labels],
    "indoor": [templates_labels, templates_indoor],
    "food": [templates_labels, templates_food],
    "svhn": [templates_svhn],
    "stl": [templates_labels]
  }
  
  def __init__(self):
    self.ds_names = list(self.captions_templates.keys())
  
  def generate_captions(self, ds, ds_name):
    if ds_name not in self.ds_names:
      raise ValueError(f"{ds_name} not in {self.ds_names}")
    
    temps = []
    for temp in self.captions_templates[ds_name]:
      temps.extend(temp)
    
    # each ds: {<id>: {"path": <path>, "label": [<label(s)>]}}
    captions = {}
    temps_ordered = np.random.randint(low = 0, high = len(temps), size = (len(ds)))
    for i,k in enumerate(ds):
      lbs_string = ", ".join(ds[k]["label"])
      cap = temps[temps_ordered[i]].format(lbs_string)
      cap = re.sub(r"\s+", " ", cap).strip().lower()
      captions[ds_name + "_" + str(k)] = {
        "path": ds[k]["path"],
        "caption": cap
      }
    return captions


# ---- Script
if __name__ == "__main__":
  print("-"*70 + "\n:: Loading COCO dataset")
  coco_train, coco_droppped_train = get_coco_captions("../coco/annotations/captions_train2017.json")
  coco_val, coco_droppped_val = get_coco_captions("../coco/annotations/captions_val2017.json")

  print("-"*70 + "\n:: Loading Visual Genome dataset")
  genome_captions, dropped_genome = get_genome_captions()

  print("-"*70 + "\n:: Loading Flickr30k dataset")
  captions_flickr, dropped_flickr = get_flickr30k_captions()

  print("-"*70 + "\n:: Loading OpenImages Dataset")
  open_images_img2lab_val, oi_dropped_val = get_open_images_labels(
    "../downsampled-open-images-v4/validation-annotations-human-imagelabels-boxable.csv"
  )
  open_images_img2lab_train, oi_dropped_train = get_open_images_labels(
      "../downsampled-open-images-v4/train-annotations-human-imagelabels-boxable.csv"
  )
  open_images_img2lab_test, oi_dropped_test = get_open_images_labels(
      "../downsampled-open-images-v4/test-annotations-human-imagelabels-boxable.csv"
  )

  print("-"*70 + "\n:: Loading Indoor CVPR Dataset")
  img2label_indoor = get_indoor_cvpr()

  print("-"*70 + "\n:: Loading Food-101k Dataset")
  img2label_food = get_food()

  print("-"*70 + "\n:: Loading STL-10 Dataset")
  img2label_stl = get_stl10()

  print("-"*70 + "\n:: Loading SVHN Dataset")
  img2label_svhn, dropped_svhn = get_svhn_data()

  # define table for tabulate
  headers = ["name", "num_samples", "dropped"]
  table = [
    ["coco_train", len(coco_train), len(coco_droppped_train)],
    ["coco_val", len(coco_val), len(coco_droppped_val)],
    ["visual genome", len(genome_captions), len(dropped_genome)],
    ["open images (train)", len(open_images_img2lab_train), len(oi_dropped_train)],
    ["open images (val)", len(open_images_img2lab_val), len(oi_dropped_val)],
    ["open images (test)", len(open_images_img2lab_test), len(oi_dropped_test)],
    ["indoor cvpr", len(img2label_indoor), 0],
    ["food-101k", len(img2label_food), 0],
    ["STL-10", len(img2label_stl), 0],
    ["SVHN", len(img2label_svhn), len(dropped_svhn)],
  ]
  table_arr = np.asarray(table)
  total_samples = sum([len(coco_train),
                       len(coco_val),
                       len(genome_captions),
                       len(open_images_img2lab_train),
                       len(open_images_img2lab_val),
                       len(open_images_img2lab_test),
                       len(img2label_indoor),
                       len(img2label_food),
                       len(img2label_stl),
                       len(img2label_svhn)])
  total_dropped = sum([len(coco_droppped_train),
                       len(coco_droppped_val),
                       len(dropped_genome),
                       len(oi_dropped_train),
                       len(oi_dropped_val),
                       len(oi_dropped_test),
                       len(dropped_svhn)])
  table.append(["total", total_samples, total_dropped])
  print("\n", "-"*70, "\n")
  print(tabulate(table, headers, tablefmt="psql"))

  print("\nGenerating captions for labels")

  capgen = CaptionGenerator()
  capgen_oi_train = capgen.generate_captions(open_images_img2lab_train, "open_images")
  capgen_oi_val   = capgen.generate_captions(open_images_img2lab_val, "open_images")
  capgen_oi_test  = capgen.generate_captions(open_images_img2lab_test, "open_images")
  capgen_indoor   = capgen.generate_captions(img2label_indoor, "indoor")
  capgen_food     = capgen.generate_captions(img2label_food, "food")
  capgen_stl      = capgen.generate_captions(img2label_stl, "stl")
  capgen_svhn     = capgen.generate_captions(img2label_svhn, "svhn")

  # make the master captions list
  common_captions = {}
  common_captions.update(capgen_oi_train)
  common_captions.update(capgen_oi_val)
  common_captions.update(capgen_oi_test)
  common_captions.update(capgen_indoor)
  common_captions.update(capgen_food)
  common_captions.update(capgen_stl)
  common_captions.update(capgen_svhn)
  common_captions.update(coco_train)
  common_captions.update(coco_val)
  common_captions.update(genome_captions)
  common_captions.update(captions_flickr)

  assert len(common_captions) == table[-1][1]
  with open("../captions_train.json", "w") as f:
    f.write(json.dumps(common_captions))
