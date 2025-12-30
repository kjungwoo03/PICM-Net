#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation of Uncompressed Inference

Models:
    - timm (ResNet50, ConvNeXt, ViT), CLIP Vit-B/32 (zero-shot)

Command:
    - python 00_uncompressed.py --dataset "datasets/ImageNet/val" --batch_size 512 --seed 42 --save_dir results/uncompressed
"""

PROJECT_DIR = "home/jungwoo-kim/workspace/PICM-Net"
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

import os
import sys
import math
import argparse
import pandas as pd
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import timm
from tqdm import tqdm

from utils import set_seed, accuracy_topk

def arg_parse(argv):
    p = argparse.ArgumentParser("Uncompressed Inference")
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
        "--batch_size",
        type=int,
        default=256,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return p.parse_args(argv)

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
        transforms.CenterCrop((256,256)),
        transforms.ToTensor(),
    ])
    cls_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
    ])
    dataset = ImageFolder(
        root=args.dataset,
        transform=_transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
    )
    print(f"[info] Dataset size: {len(dataset) } # of batches: {len(dataloader)}")

    # load models
    # model 1. ResNet50
    models = {}
    m = timm.create_model("resnet50.a1_in1k", pretrained=True)
    m.to(device)
    m.eval()
    models["resnet50"] = m

    # model 2. ConvNeXt
    m = timm.create_model("convnext_base.fb_in1k", pretrained=True)
    m.to(device)
    m.eval()
    models["convnext"] = m

    # model 3. ViT
    m = timm.create_model("vit_large_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
    m.to(device)
    m.eval()
    models["vit_large_21k"] = m

    # inference
    results = {}  # model_name -> dict

    for model_name, model in models.items():
        print("\n====================================")
        print(f"[info] Evaluating TIMM model: {model_name}")
        print("====================================")

        total_top1 = 0.0
        total_top5 = 0.0
        total_loss = 0.0
        total_samples = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc=f"[{model_name}]")):
                images = torch.stack([cls_transform(image) for image in images])
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                logits = model(images)  # (N, 1000)
                loss = criterion(logits, targets)
                acc_dict = accuracy_topk(logits, targets, topk=(1, 5))

                total_top1 += acc_dict[1]
                total_top5 += acc_dict[5]
                total_loss += loss.item() * targets.size(0)
                total_samples += targets.size(0)

        top1_acc = total_top1 / total_samples * 100.0
        top5_acc = total_top5 / total_samples * 100.0
        avg_loss = total_loss / total_samples
        print("------------------------------------")
        print(f"[result] {model_name} | top1={top1_acc:.3f}% | top5={top5_acc:.3f}% | loss={avg_loss:.4f} | N={total_samples}")
        print("------------------------------------")

        results[model_name] = {
            "top1": top1_acc,
            "top5": top5_acc,
            "loss": avg_loss,
            "num_samples": total_samples,
        }

    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.join(args.save_dir, "results.csv"), index=False)






if __name__ == "__main__":
    args = arg_parse(sys.argv[1:])
    main(args)