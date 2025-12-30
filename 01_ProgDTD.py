#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Implementation of ProgDTD

Source:
    - https://github.com/ds-kiel/ProgDTD.git
Command:
    - python 01_ProgDTD.py --model_path "pretrained_weights/ProgDTD/lambda0.1-range[0.0-1.0].pth" --dataset "datasets/ImageNet/val" --save_dir "results/01_ProgDTD/lambda0.1-range[0.0-1.0]" --seed 42 --batch_size 512
'''

PROJECT_DIR = "/home/jungwoo-kim/workspace/PICM-Net"
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

from compressai.layers import GDN as GeneralizedDivisiveNormalization
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

import timm

from utils import set_seed, clean_progdtd_state_dict, eval_metrics, accuracy_topk, CLIPZeroShotWrapper

def _conv( cin, cout, kernel_size, stride=1) -> nn.Conv2d:
    return nn.Conv2d(
        cin, cout, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
    )

def _deconv(cin, cout, kernel_size, stride = 1) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        cin,
        cout,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class _AbsoluteValue(nn.Module):
    def forward(self, inp: Tensor) -> Tensor:
        return torch.abs(inp)
     
class ImageAnalysis(nn.Module):
    def __init__(self, network_channels: int, compression_channels: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            _conv(3, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels),
            _conv(network_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels),
            _conv(network_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels),
            _conv(network_channels, compression_channels, kernel_size=5, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
class ImageSynthesis(nn.Module): 
    def __init__(self, network_channels: int, compression_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            _deconv(compression_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels, inverse=True),
            _deconv(network_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels, inverse=True),
            _deconv(network_channels, network_channels, kernel_size=5, stride=2),
            GeneralizedDivisiveNormalization(network_channels, inverse=True),
            _deconv(network_channels, 3, kernel_size=5, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
class HyperAnalysis(nn.Module):
    def __init__(self, network_channels: int, compression_channels: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            _AbsoluteValue(),
            _conv(compression_channels, network_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            _conv(network_channels, network_channels, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            _conv(network_channels, network_channels, kernel_size=5, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
class HyperSynthesis(nn.Module):
    def __init__(self, network_channels: int, compression_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            _deconv(network_channels, network_channels, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            _deconv(network_channels, network_channels, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            _deconv(network_channels, compression_channels, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class ScaleHyperprior(nn.Module):
    def __init__(
        self,
        network_channels: Optional[int] = None,
        compression_channels: Optional[int] = None,
        image_analysis: Optional[nn.Module] = None,
        image_synthesis: Optional[nn.Module] = None,
        image_bottleneck: Optional[nn.Module] = None,
        hyper_analysis: Optional[nn.Module] = None,
        hyper_synthesis: Optional[nn.Module] = None,
        hyper_bottleneck: Optional[nn.Module] = None,
        progressiveness_range: Optional[List] = None,
    ):
        super().__init__()
        self.image_analysis = ImageAnalysis(network_channels, compression_channels)  
        self.hyper_analysis = HyperAnalysis(network_channels, compression_channels) 
        self.hyper_synthesis = HyperSynthesis(network_channels, compression_channels)  
        self.image_synthesis = ImageSynthesis(network_channels, compression_channels)
        
        self.hyper_bottleneck = EntropyBottleneck(channels=network_channels)
        self.image_bottleneck = GaussianConditional(scale_table=None)
        self.progressiveness_range = progressiveness_range
        self.p_hyper_latent = None
        self.p_latent = None
        
    def forward(self, images):
            
        self.latent = self.image_analysis(images)
        self.hyper_latent = self.hyper_analysis(self.latent)
        
        #---***---#
        self.latent = self.rate_less_latent(self.latent)
        self.hyper_latent = self.rate_less_hyper_latent(self.hyper_latent)
        #---***---#

        
        self.noisy_hyper_latent, self.hyper_latent_likelihoods = self.hyper_bottleneck(
            self.hyper_latent
        )

        self.scales = self.hyper_synthesis(self.noisy_hyper_latent)
        self.noisy_latent, self.latent_likelihoods = self.image_bottleneck(self.latent, self.scales)
        
        #---***---#
        self.latent_likelihoods = self.drop_zeros_likelihood(self.latent_likelihoods, self.latent)
        self.hyper_latent_likelihoods = self.drop_zeros_likelihood(self.hyper_latent_likelihoods, self.hyper_latent)
        #---***---#
        
        self.reconstruction = self.image_synthesis(self.noisy_latent)

        self.rec_image = self.reconstruction.detach().clone()

        return self.reconstruction, self.latent_likelihoods, self.hyper_latent_likelihoods

    def rate_less_latent(self, data):
        self.save_p = []
        temp_data = data.clone()
        for i in range(data.shape[0]):
            if self.p_latent:
                # p shows the percentage of keeping
                p = self.p_latent
            else:
                p = np.random.uniform(self.progressiveness_range[0], self.progressiveness_range[1],1)[0]
                self.save_p.append(p)

            if p == 1.0:
                pass            
            else:
                p = int(p*data.shape[1])
                replace_tensor = torch.rand(data.shape[1]-p-1, data.shape[2], data.shape[3]).fill_(0)

                if replace_tensor.shape[0] > 0:
                    temp_data[i,-replace_tensor.shape[0]:,:,:] =  replace_tensor
                    
        return temp_data
    
    def rate_less_hyper_latent(self, data):
        temp_data = data.clone()
        for i in range(data.shape[0]):
            if self.p_hyper_latent:
                # p shows the percentage of keeping
                p = self.p_hyper_latent
            else:
                p = np.random.uniform(self.progressiveness_range[0], self.progressiveness_range[1], 1)[0]
                p = self.save_p[i]
            if p == 1.0:
                pass
            
            else:
                p = int(p*data.shape[1])
                replace_tensor = torch.rand(data.shape[1]-p-1, data.shape[2], data.shape[3]).fill_(0)

                if replace_tensor.shape[0] > 0:
                    temp_data[i,-replace_tensor.shape[0]:,:,:] =  replace_tensor
                    
        return temp_data

    def drop_zeros_likelihood(self, likelihood, replace):
        temp_data = likelihood.clone()
        temp_data = torch.where(
            replace == 0.0,
            torch.cuda.FloatTensor([1.0])[0],
            likelihood,
        )
        return temp_data
    
def arg_parse(argv):
    p = argparse.ArgumentParser("ProgDTD Inference")
    p.add_argument(
        "--model_path", 
        type=str, 
        required=True,
    )
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
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
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
    
    # load model
    model = ScaleHyperprior(
        network_channels=128,
        compression_channels=192,
        progressiveness_range=[0.3, 1.0],
    ).to(device)
    state_dict = clean_progdtd_state_dict(torch.load(args.model_path, map_location="cpu"), verbose=False)
    model.load_state_dict(state_dict, strict=False)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[info] Loaded model: {args.model_path} | Parameters: {total_params / 1e6:.2f}M")

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

    # model 4. CLIP
    imagenet_classnames_path = os.path.join(os.path.dirname(args.dataset), "classnames.txt")
    clip_cache = os.path.join(PROJECT_DIR, "results/00a_clip_zeroshot/text_embeds.npy")
    clip_wrapper = CLIPZeroShotWrapper(
        model_id="openai/clip-vit-large-patch14",
        device=device,
        classnames_path=imagenet_classnames_path,
        dataset_classes=dataset.classes,
        text_embed_cache=clip_cache,
    )
    models["clip"] = clip_wrapper

    # inference
    model.eval()
    # p_list = [50]
    p_list = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100]
    p_list.reverse()
    results = {}
    for p_int in p_list:
        p = p_int / 100.0
        model.p_latent = p
        model.p_hyper_latent = p

        print("\n====================================")
        print(f"[info] Evaluating ProgDTD at p={p:.2f}")
        print("====================================")

        # codec metrics
        bpp_list: List[float] = []
        mse_list: List[float] = []
        psnr_list: List[float] = []
        ssim_list: List[float] = []
        msssim_list: List[float] = []

        # classifier metrics (per model)
        cls_stats: Dict[str, Dict[str, float]] = {}
        for name in models.keys():
            cls_stats[name] = {
                "top1": 0.0,
                "top5": 0.0,
                "loss": 0.0,
                "num_samples": 0.0,
            }

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc=f"[p={p:.2f}]")):
                # images: (N, 3, 256, 256)
                N, _, H, W = images.shape
                num_pixels = float(N * H * W)

                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # codec forward
                x_hat, y_likelihoods, z_likelihoods = model(images)
                x_hat = x_hat.clamp(0.0, 1.0)

                # bpp calculation
                bits = (y_likelihoods.log().sum() + z_likelihoods.log().sum()) / (-math.log(2.0))
                bpp = (bits / num_pixels).item()
                bpp_list.append(bpp)

                # human RD metrics
                mse, psnr, ssim, msssim = eval_metrics(images, x_hat, verbose=False)
                mse_list.append(mse)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                msssim_list.append(msssim)

                # prepare machine inference
                x_hat_cpu = x_hat.detach().cpu()
                cls_batch = torch.stack([cls_transform(img) for img in x_hat_cpu])
                cls_batch = cls_batch.to(device, non_blocking=True)

                # evaluate machine metrics
                for model_name, cls_model in models.items():
                    if model_name == "clip":
                        logits = cls_model(x_hat)
                    else:
                        logits = cls_model(cls_batch)

                    acc_dict = accuracy_topk(logits, targets, topk=(1, 5))
                    cls_stats[model_name]["top1"] += acc_dict[1]
                    cls_stats[model_name]["top5"] += acc_dict[5]
                    cls_stats[model_name]["loss"] += acc_dict['ce_loss']
                    cls_stats[model_name]["num_samples"] += float(targets.size(0))

        # aggregate metrics for this p
        mean_bpp = float(np.mean(bpp_list))
        mean_mse = float(np.mean(mse_list))
        mean_psnr = float(np.mean(psnr_list))
        mean_ssim = float(np.mean(ssim_list))
        mean_msssim = float(np.mean(msssim_list))

        if args.verbose:
            print("------------------------------------")
            print(f"[codec] p={p:.2f} | bpp={mean_bpp:.4f} | mse={mean_mse:.4f} | "
                  f"psnr={mean_psnr:.3f} | ssim={mean_ssim:.4f} | msssim={mean_msssim:.4f}")

        # results row for this p
        row: Dict[str, float] = {
            "p": p,
            "bpp": mean_bpp,
            "mse": mean_mse,
            "psnr": mean_psnr,
            "ssim": mean_ssim,
            "msssim": mean_msssim,
        }

        # add classifier metrics
        for model_name, stat in cls_stats.items():
            n = max(1.0, stat["num_samples"])
            top1 = stat["top1"] / n * 100.0
            top5 = stat["top5"] / n * 100.0
            avg_loss = stat["loss"] / n

            if args.verbose:
                print(f"[{model_name}] p={p:.2f} | top1={top1:.3f}% | top5={top5:.3f}% | "
                      f"loss={avg_loss:.4f} | N={int(stat['num_samples'])}")

            row[f"{model_name}_top1"] = top1
            row[f"{model_name}_top5"] = top5
            row[f"{model_name}_loss"] = avg_loss
            row[f"{model_name}_N"] = stat["num_samples"]

        results[f"p={p:.2f}"] = row

    # Save results
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "p_key"
    out_path = os.path.join(args.save_dir, "results.csv")
    df.to_csv(out_path)
    print(f"[info] Saved results to: {out_path}")


if __name__ == "__main__":
    args = arg_parse(sys.argv[1:])
    main(args)