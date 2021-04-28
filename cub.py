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

from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE1024, DiscreteVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer

from tokenizers import Tokenizer


BPE_path = "./saved_model/tokenizers/pr_bpe.json"
vocab = Tokenizer.from_file(BPE_path)


def show(img):
    npimg = img.numpy()
    plt.figure(figsize=(100, 40))
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


def preprocess_data(caption, imagepath):
    image = Image.open(imagepath)
    image_size = 256
    image_tranform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.RandomResizedCrop(image_size, scale=(0.75, 1.), ratio=(1., 1.)),
        T.ToTensor()
    ])

    input_text = [caption]
    token_list = []
    sot_token = vocab.encode("<|startoftext|>").ids[0]
    eot_token = vocab.encode("<|endoftext|>").ids[0]
    for txt in input_text:
        codes = [0] * 80
        text_token = vocab.encode(txt).ids
        tokens = [sot_token] + text_token + [eot_token]
        codes[:len(tokens)] = tokens
        caption_token = torch.LongTensor(codes).cuda()
        token_list.append(caption_token)
    text = torch.stack(token_list)
    mask = (text != 0).cuda()

    image_tensor = image_tranform(image)

    return (text.unsqeeze(0), image_tensor.unsqueeze(0), mask.unsqueeze(0))


def get_first_line(filepath):
    f = open(filepath,'r')
    line = f.readline()
    f.close()
    return line

vae_dict = {"args": {"image_size": 256, "emb_dim": 256}}
vae = VQGanVAE1024()

if (os.path.exists("./saved_models/dalle_cub.pth")):
    dalle_pr_dict = torch.load("./saved_models/dalle_cub.pth")
    attn_types = []
    for type in dalle_pr_dict['args']['attn_types'].split(","):
        assert type in ("full", "sparse", "axial_row", "axial_col", "conv_like")
        attn_types.append(type)
    attn_types = tuple(attn_types)
    dalle = DALLE(
        dim=vae_dict['args']['emb_dim'],
        vae=vae,
        num_text_tokens=dalle_pr_dict['args']['num_text_tokens'],
        text_seq_len=dalle_pr_dict['args']['text_seq_len'],
        depth=dalle_pr_dict['args']['depth'],
        heads=dalle_pr_dict['args']['heads'],
        reversible=dalle_pr_dict['args']['reversible'],
        attn_types=attn_types
    ).cuda()

else:
    dalle = DALLE(
        dim=1024,
        # automatically infer (1) image sequence length and (2) number of image tokens
        vae=vae,
        num_text_tokens=7800,    # vocab size for text
        text_seq_len=256,         # text sequence length
        depth=1,                  # should aim to be 64
        heads=16,                 # attention heads
        dim_head=64,              # attention head dimension
        attn_dropout=0.1,         # attention dropout
        ff_dropout=0.1            # feedforward dropout
    ).cuda()


cub_dir = "./CUB_200_2011/images/"
cub_cap_dir = "./birds/text_c10/"
image_paths = [cub_dir + file[1] + file[2] for file in os.walk(cub_dir)]
captions = [get_first_line(cub_cap_dir + file[1] + file[2]) for file in os.walk(cub_cap_dir)]


EPOCHS = 20
LEARNING_RATE = 3e-4
LR_DECAY_RATE = 0.98
# BATCH_SIZE = 2
GRAD_CLIP_NORM = 0.5

opt = Adam(dalle.parameters(), lr = LEARNING_RATE)
sched = ExponentialLR(optimizer = opt, gamma = LR_DECAY_RATE)

for epoch in range(1,EPOCHS+1):
  i = 0
  for imagepath,caption in zip(image_paths,captions):
    i+=1
    # for i in range(BATCH_SIZE):
    text,image,mask = preprocess_data(caption,imagepath)

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
    #   torch.save(dalle,"./saved_models/dalle_cub.pt") # Uncomment to save model at directory

# torch.save(dalle,"./saved_models/dalle_cub.pt") # Uncomment to save model at directory

input_text = ["Red bird"] * 4

token_list = []
sot_token = vocab.encode("<|startoftext|>").ids[0]
eot_token = vocab.encode("<|endoftext|>").ids[0]
for txt in input_text:
    codes = [0] * dalle_pr_dict['args']['text_seq_len']
    text_token = vocab.encode(txt).ids
    tokens = [sot_token] + text_token + [eot_token]
    codes[:len(tokens)] = tokens
    caption_token = torch.LongTensor(codes).cuda()
    token_list.append(caption_token)
text = torch.stack(token_list)
mask = (text != 0).cuda()

images = dalle.generate_images(text, mask = mask, filter_thres = 0.9, temperature=1.0)

grid = make_grid(images, nrow=4, normalize=False, range=(-1, 1)).cpu()
show(grid)

save_image(images[1],'images/test_red_bird.jpg', normalize=True)