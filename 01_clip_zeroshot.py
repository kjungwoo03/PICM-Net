#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Implementation of CLIP Zero-shot Inference

Command:
    - python 01_clip_zeroshot.py --dataset "datasets/ImageNet/val" --save_dir "results/01_clip_zeroshot" --seed 42 --batch_size 512
'''


PROJECT_DIR = "home/jungwoo-kim/workspace/PICM-Net"
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

import os
import sys
import math
import argparse
import pickle
import glob
import yaml
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import PIL 
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import ImageFolder

from transformers import CLIPModel, CLIPProcessor

from compressai.zoo import mbt2018_mean
from compressai.layers import GDN as GeneralizedDivisiveNormalization
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

import timm

from utils import set_seed, eval_metrics, accuracy_topk, clear_cache


def arg_parse(argv):
    p = argparse.ArgumentParser("CLIP Zero-shot Inference")
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    p.add_argument(
        "--save_dir",
        type=str,
        required=True,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    return p.parse_args(argv)



def extract_embeds(model, processor, imagenet_classes, text_embed_path, device):
    prompts = [f"a photo of a {imagenet_classes[i]}" for i in range(len(imagenet_classes))]
    
    with torch.no_grad():
        inputs = processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        text_embeds = model.get_text_features(**inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # L2 normalize
    
    np.save(text_embed_path, text_embeds.cpu().numpy())
    print(f"[info] Saved text embeddings to {text_embed_path}, shape: {text_embeds.shape}")
    return text_embeds


def main(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device} | Number of GPUs: {torch.cuda.device_count()}")

    # set seed
    if args.seed is not None:
        set_seed(args.seed)
        print(f"[info] Set seed to {args.seed}")

    # check save dir
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[info] Save directory: {args.save_dir}")

    # dataset load
    _transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(
        root=args.dataset,
        transform=_transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    print(f"[info] Dataset size: {len(dataset)} | # of batches: {len(dataloader)}")

    # CLIP model
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14"
    ).to(device)
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14",
        use_fast=True  # Explicitly use the fast processor
    )
    model.eval()

    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std
    crop_size = processor.image_processor.crop_size["height"]  # 224
    resize_size = processor.image_processor.size["shortest_edge"]  # 224 or 256 (모델마다 다를 수 있음)

    clip_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])
        

    # ImageNet Classes
    imagenet_classnames_path = os.path.join(os.path.dirname(args.dataset), "classnames.txt")

    # 1) synset -> human-readable name 매핑 만들기
    synset_to_name: Dict[str, str] = {}
    with open(imagenet_classnames_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                synset = parts[0]           # e.g. n02119789
                # idx = int(parts[1])       # 이건 이제 안 써도 됨
                class_name = " ".join(parts[2:])
                synset_to_name[synset] = class_name

    print(f"[info] Loaded {len(synset_to_name)} ImageNet classes from classnames.txt")

    # 2) ImageFolder의 클래스 순서에 맞춰 이름 리스트 만들기
    #    dataset.classes: ['n01440764', 'n01443537', ...]  (알파벳 순서)
    idx_to_name: List[str] = []
    for synset in dataset.classes:
        if synset not in synset_to_name:
            raise ValueError(f"Synset {synset} not found in classnames.txt")
        idx_to_name.append(synset_to_name[synset])

    print(f"[info] Dataset #classes: {len(idx_to_name)}")

    # 3) 프롬프트 생성
    prompts = [f"a photo of a {name.replace('_', ' ')}" for name in idx_to_name]
    print(f"[info] Example prompt[0]: {prompts[0]}")

    # 4) 텍스트 임베딩 추출 함수도 prompts를 받게 수정
    def extract_embeds(model, processor, prompts, text_embed_path, device):
        with torch.no_grad():
            inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            text_embeds = model.get_text_features(**inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        np.save(text_embed_path, text_embeds.cpu().numpy())
        print(f"[info] Saved text embeddings to {text_embed_path}, shape: {text_embeds.shape}")
        return text_embeds

    # 5) 호출 부분
    text_embed_path = os.path.join(args.save_dir, "text_embeds.npy")
    if os.path.exists(text_embed_path):
        text_embeds = torch.from_numpy(np.load(text_embed_path)).to(device)
        print(f"[info] Loaded text embeddings from {text_embed_path}, shape: {text_embeds.shape}")
    else:
        text_embeds = extract_embeds(
            model, processor, prompts, text_embed_path, device
        )

     # inference
    model.eval()
    total_top1 = 0
    total_top5 = 0
    total_samples = 0
    total_ce = 0.0  # CE 합계

    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Inference")):
        images = torch.stack([clip_transform(image) for image in images])
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            image_embeds = model.get_image_features(pixel_values=images)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            # logits: (B, num_classes)
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * (image_embeds @ text_embeds.T)

            # ----- Cross-Entropy 계산 (natural log base) -----
            # reduction='sum'으로 해서 나중에 전체 샘플 수로 나누기
            ce = F.cross_entropy(logits_per_image, targets, reduction='sum')
            total_ce += ce.item()

            # ----- Accuracy 계산 -----
            probs = logits_per_image.softmax(dim=1)
            _, pred_top5 = probs.topk(5, dim=1)

            correct_top1 = (pred_top5[:, 0] == targets).sum().item()
            correct_top5 = sum(
                targets[i].item() in pred_top5[i].cpu().tolist()
                for i in range(targets.size(0))
            )

            total_top1 += correct_top1
            total_top5 += correct_top5
            total_samples += targets.size(0)

    top1_acc = total_top1 / total_samples * 100.0
    top5_acc = total_top5 / total_samples * 100.0
    avg_ce = total_ce / total_samples  # per-sample cross-entropy

    print(f"[result] CLIP | top1={top1_acc:.3f}% | top5={top5_acc:.3f}% | CE={avg_ce:.4f} | N={total_samples}")

    


if __name__ == "__main__":
    args = arg_parse(sys.argv[1:])
    main(args)