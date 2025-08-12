import os
import torch
import clip
import random as rd
import numpy as np
import pandas as pd
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from model.camera_utils import get_camera_org

dist = 2
elev_azi = [    
    (dist, 0, 180), # front
    (dist, 0, 0), # back
    (dist, 0, 90), # left
    (dist, 0, 270), # right
    (dist, 90, 0), # up
    (dist, 270, 0) # bottom
]

class QACollator:
    def __init__(self):

        self.tokenizer_en=clip.tokenize
        tokenizer_zh=AutoTokenizer.from_pretrained("bert-base-chinese")

        self.tokenizer_zh = lambda x: tokenizer_zh(
                x,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

    def __call__(self, batch):
        images, d_images, order, prompt, prompt_zh, mos = zip(*batch)

        images = torch.stack(images)
        d_images = torch.stack(d_images)
        order = torch.stack(order)
        mos = torch.stack(mos)

        prompt = self.tokenizer_en(prompt)
        prompt_zh = self.tokenizer_zh(prompt_zh)

        return images, d_images, order, prompt, prompt_zh, mos


class Mate3DData(Dataset):
    def __init__(self, infos, data_dir, transforms=None, shuffle=False):
        super().__init__()

        self.data_dir = data_dir
        self.info = infos
        self.transforms = transforms

        self.num_views = 6
        self.shuffle = shuffle
        
        self.cameras = get_camera_org(
            elev_azi_pairs=elev_azi
        )
        
        self.transforms_depth = T.Compose([
            T.Resize(224),
            T.ToTensor()
        ])

        chinese_data_info = pd.read_excel("/home/lhh/codes/AIG3DCQA/OBenchmark/data/csvfiles/prompts_MATE_3D.xlsx").to_dict("records")

        self.chinese_prompt_dict = {}

        for d in chinese_data_info:
            self.chinese_prompt_dict[d["Prompt"].strip()] = d["Chinese Prompt"]

    def __len__(self):
        return len(self.info)
    
    def __getitem__(self, index):
        
        info = self.info[index]
        
        # mid = info["id"]
        geo_mos = info["Geometry"]
        texture_mos = info["Texture"]
        align_mos = info["Alignment"]
        overall_mos = info["Overall"]

        prompt = info["prompt"]
        prompt_zh = self.chinese_prompt_dict.get(prompt, prompt)
        prompt_zh = prompt
        name_prompt = prompt.replace(" ", "_")
        name_model = info["model"]
        
        images = []
        d_images = []
        
        for i in range(self.num_views):
            
            # imge_name = os.path.join(self.data_dir, f"{mid}_{i}_orthographic.png")
            dimge_name = os.path.join(self.data_dir, f"{name_model}_{name_prompt}_{i}_depth_orthographic.png")
            imge_name = os.path.join(self.data_dir, f"{name_model}_{name_prompt}_{i}_orthographic.png")
            if os.path.exists(imge_name):
                img = Image.open(imge_name)
                img = img.convert('RGB')
                img = self.transforms(img)
                images.append(img)
                
            if os.path.exists(dimge_name):
                dimg = Image.open(dimge_name)
                dimg = dimg.convert('L')
                # img = self.transforms(img)
                dimg = self.transforms_depth(dimg).expand(3, -1, -1)
                d_images.append(dimg)
        
        images = torch.stack(images, dim=0)
        d_images = torch.stack(d_images, dim=0)
        # d_images = images.clone()
        
        if self.shuffle:
            order = rd.sample(range(0, self.num_views), self.num_views)
            images = images[order,:,:,:]
            d_images = d_images[order,:,:,:]
        else:
            order = list(range(self.num_views))
        
        mos = torch.tensor([
            geo_mos,
            texture_mos,
            align_mos,
            overall_mos
        ], dtype=torch.float32) / 2
        
        order = torch.tensor(order, dtype=torch.long)

        return images, d_images, order, prompt, prompt_zh, mos
    
    
    def get_pair(self, idx):
        
        info = self.info[idx]
        
        geo_mos = info["Geometry"]
        texture_mos = info["Texture"]
        align_mos = info["Alignment"]
        overall_mos = info["Overall"]

        prompt = info["prompt"]
        name_prompt = prompt.replace(" ", "_")
        name_model = info["model"]
        
        samples = [[0, 1], [2, 3], [4, 5]]
        
        sample = [rd.choice(s) for s in samples]
        sample = rd.choices(sample, k=2)
        
        
        
        