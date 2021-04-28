# DISCLAIMER: Code at this state will not work due to 
# missing libraries, dataset files and pretrained models
# (due to size constraints of github).
# Also we used Google Colab to run the code, and we copied 
# over the contents, but we might have missed something while copying
# Please refer to README to get the data and pretrained models

import pandas as pd
# from google.colab import drive
import os
import requests
from io import BytesIO
import numpy as np 
from PIL import Image
from random import choice
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

from dalle_pytorch import OpenAIDiscreteVAE,VQGanVAE1024, DiscreteVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer

def show(img):
  npimg = img.numpy()
  plt.figure(figsize = (100,40))
  plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def preprocess_data(text,imagepath):
  image = Image.open(imagepath)
  image_size = 256
  image_tranform = T.Compose([
      T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
      T.RandomResizedCrop(image_size, scale = (0.75, 1.), ratio = (1., 1.)),
      T.ToTensor()
  ])

  tokenized_text = tokenizer.tokenize(text)
  mask = tokenized_text != 0
  image_tensor = image_tranform(image)

  return (tokenized_text,image_tensor.unsqueeze(0),mask)


flower_dir = "" # insert flower directory here (Refer to readme for dataset link)
flower_filenames = [file[2] for file in os.walk(flower_dir)][0]
flower_filepaths = [flower_dir+name for name in flower_filenames]

vae = OpenAIDiscreteVAE().cuda()

if (os.path.exists("./saved_models/dalle_flower.pt")):
  dalle = torch.load("./saved_models/dalle_flower.pt")
  dalle = dalle.cuda()
  dalle.eval()
else:
  dalle = DALLE(
      dim = 1024,
      vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
      num_text_tokens = 49408,    # vocab size for text
      text_seq_len = 256,         # text sequence length
      depth = 1,                  # should aim to be 64
      heads = 16,                 # attention heads
      dim_head = 64,              # attention head dimension
      attn_dropout = 0.1,         # attention dropout
      ff_dropout = 0.1            # feedforward dropout
  ).cuda()

EPOCHS = 20
LEARNING_RATE = 3e-4
LR_DECAY_RATE = 0.98
# BATCH_SIZE = 2
GRAD_CLIP_NORM = 0.5

opt = Adam(dalle.parameters(), lr = LEARNING_RATE)
sched = ExponentialLR(optimizer = opt, gamma = LR_DECAY_RATE)

for epoch in range(1,EPOCHS+1):
  i = 0
  for imagepath,imagename in zip(flower_filepaths,flower_filenames):
    i+=1
    # for i in range(BATCH_SIZE):
    text,image,mask = preprocess_data(imagename,imagepath)

    if (text == None and image == None):
      continue

    loss = dalle(text.cuda(), image.cuda(), mask = mask.cuda(), return_loss = True)

    loss.backward()
    clip_grad_norm_(dalle.parameters(), GRAD_CLIP_NORM)

    opt.step()
    opt.zero_grad()

    if i % 10 == 0:
      print(epoch, i, f'loss - {loss.item()}')
    
    # if i % 100 == 0:
    #   torch.save(dalle,"./saved_models/dalle_flower.pt") # Uncomment to save model at directory

# torch.save(dalle,"./saved_models/dalle_flower.pt") # Uncomment to save model at directory

input_text = ["lilies"] * 4 # 4 for 4 image generations

text = tokenizer.tokenize(input_text).cuda()
mask = text != 0

images = dalle.generate_images(text, mask = mask.cuda(), filter_thres = 0.9, temperature=1.0)

grid = make_grid(images, nrow=4, normalize=False, range=(-1, 1)).cpu()
show(grid)

save_image(images[3],'images/test_lilies.jpg', normalize=True)