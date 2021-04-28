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


train_data = pd.read_table("/content/drive/My Drive/DALLE_CS390_data/Train_GCC-training.tsv")
test_data = pd.read_table("/content/drive/My Drive/DALLE_CS390_data/Validation_GCC-1.1.0-Validation.tsv")

train_data.columns = ["caption","url"]
test_data.columns = ["caption", "url"]


def get_image_from_url(image_url):
  caption,url = row
  # print(url)
  try:
    res = requests.get(url,timeout=(2, 5))
    if res.status_code == 200:
        img_arr = Image.open(BytesIO(res.content))
        # print(img_arr.shape)
        return img_arr
    else:
      return None
  except:
    return None
def preprocess_data(text,image_url):
  image = get_image_from_url(image_url)
  image_size = 256
  if (image == None):
    return None,None,None 

  image_tranform = T.Compose([
      T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
      T.RandomResizedCrop(image_size, scale = (0.75, 1.), ratio = (1., 1.)),
      T.ToTensor()
  ])

  tokenized_text = tokenizer.tokenize(text)
  mask = tokenized_text != 0
  image_tensor = image_tranform(image)

  return (tokenized_text,image_tensor.unsqueeze(0),mask.unsqueeze(0))

vae = OpenAIDiscreteVAE().cuda()      # loads pretrained OpenAI VAE

dalle = None

res_train_data = train_data


if (os.path.exists("./saved_models/dalle_gen.pt")):
  dalle = torch.load("./saved_models/dalle_gen.pt")
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

EPOCHS = 5
LEARNING_RATE = 3e-4
LR_DECAY_RATE = 0.98
# BATCH_SIZE = 2
GRAD_CLIP_NORM = 0.5

opt = Adam(dalle.parameters(), lr = LEARNING_RATE)
sched = ExponentialLR(optimizer = opt, gamma = LR_DECAY_RATE)

for epoch in range(1,EPOCHS+1):
  for i,row in res_train_data.iterrows():
    caption,img_url = row

    # for i in range(BATCH_SIZE):
    text,image,mask = preprocess_data(caption,img_url)

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
    #   torch.save(dalle,"./saved_models/dalle_gen.pt")
  
# torch.save(dalle,"./saved_models/dalle_gen.pt")

s_text = "taking a plunge into river"
s_text_tok = tokenizer.tokenize(s_text).cuda()
s_mask = s_text_tok != 0
images = dalle.generate_images(s_text_tok, mask = s_mask.cuda())
save_image(images[0],'images/test.jpg', normalize=True)