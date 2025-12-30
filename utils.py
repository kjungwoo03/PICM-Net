#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Util functuions
'''

import gc
import torch
import random
from pytorch_msssim import ssim, ms_ssim
import torch.nn.functional as F
import clip
from transformers import CLIPModel, CLIPProcessor
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import os
import numpy as np
from typing import List, Dict, Optional
from torch import Tensor, nn

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def clip_inference(images, model):
    images = images.to(model.device)
    images = images.unsqueeze(0)
    images = images.to(model.device)
    
def clean_progdtd_state_dict(raw_state, verbose=False):

    if not isinstance(raw_state, dict):
        raise TypeError(f"raw_state must be dict, got {type(raw_state)}")

    cleaned_state = {}
    for k, v in raw_state.items():
        if not isinstance(v, torch.Tensor):
            continue

        if k.startswith("model."):
            new_k = k[len("model."):]  # remove 'model.' prefix
            cleaned_state[new_k] = v
        else:
            if (
                k.startswith("image_analysis.") or
                k.startswith("hyper_analysis.") or
                k.startswith("hyper_synthesis.") or
                k.startswith("image_synthesis.") or
                k.startswith("hyper_bottleneck.") or
                k.startswith("image_bottleneck.")
            ):
                cleaned_state[k] = v

    if verbose:
        print(f"[info] clean_progdtd_state_dict: in={len(raw_state)} keys, out={len(cleaned_state)} keys")

        some_keys = list(cleaned_state.keys())[:10]
        print("[info] example cleaned keys:", some_keys)

    return cleaned_state

def PSNR(x, x_hat):
    """Calculate PSNR between two images (higher is better)"""
    mse = (x - x_hat).pow(2).mean()
    return 10 * torch.log10(1.0 / mse)

def SSIM(x, x_hat):
    """Calculate SSIM between two images (higher is better, range [0, 1])"""
    return ssim(x, x_hat, data_range=1.0, size_average=True)

def MS_SSIM(x, x_hat):
    """Calculate MS-SSIM between two images (higher is better, range [0, 1])"""
    return ms_ssim(x, x_hat, data_range=1.0, size_average=True)

def eval_metrics(x, x_hat, verbose=False):
    # Check if input is batched (4D) or single image (3D)
    if x.dim() == 4:  # Batched input [B, C, H, W]
        batch_size = x.size(0)
        
        # Calculate metrics for each image in the batch
        mse_list = []
        psnr_list = []
        ssim_list = []
        msssim_list = []
        
        for i in range(batch_size):
            x_single = x[i:i+1]  # Keep batch dimension for consistency
            x_hat_single = x_hat[i:i+1]
            
            mse_single = (x_single - x_hat_single).pow(2).mean()
            psnr_single = PSNR(x_single, x_hat_single)
            ssim_single = SSIM(x_single, x_hat_single)
            msssim_single = MS_SSIM(x_single, x_hat_single)
            
            mse_list.append(mse_single)
            psnr_list.append(psnr_single)
            ssim_list.append(ssim_single)
            msssim_list.append(msssim_single)
        
        # Average across batch
        mse = torch.stack(mse_list).mean()
        psnr = torch.stack(psnr_list).mean()
        ssim_val = torch.stack(ssim_list).mean()
        msssim = torch.stack(msssim_list).mean()
        
    elif x.dim() == 3:  # Single image [C, H, W]
        # Add batch dimension for metric calculation
        x = x.unsqueeze(0)
        x_hat = x_hat.unsqueeze(0)
        
        mse = (x - x_hat).pow(2).mean()
        psnr = PSNR(x, x_hat)
        ssim_val = SSIM(x, x_hat)
        msssim = MS_SSIM(x, x_hat)
        
    else:
        raise ValueError(f"Input tensor must be 3D [C, H, W] or 4D [B, C, H, W], got {x.dim()}D")

    if verbose:
        print(f"[info] eval_metrics: MSE={mse:.4f} | PSNR={psnr:.4f} | SSIM={ssim_val:.4f} | MS-SSIM={msssim:.4f}")

    return mse.item(), psnr.item(), ssim_val.item(), msssim.item()

def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    """
    logits: (N, C)
    targets: (N,)
    return: dict {1: correct_top1, 5: correct_top5, 'ce_loss': ce_loss} (float 개수)
    """
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, 1, True, True)  # (N, maxk)
    pred = pred.t()  # (maxk, N)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))  # (maxk, N)

    res = {}
    for k in topk:
        # 앞에서 k개 줄(= top-k)만 가져와서 flatten 후 합산
        correct_k = correct[:k].reshape(-1).float().sum().item()
        res[k] = correct_k
    
    # CE loss 계산
    ce_loss = F.cross_entropy(logits, targets).item()
    res['ce_loss'] = ce_loss
    
    return res

class CLIPZeroShotWrapper(nn.Module):
    """
    Zero-shot CLIP 분류기.
    - 입력: x (N,3,H,W), [0,1]
    - 출력: logits (N, num_classes)  (image_embeds @ text_embeds^T)
    """
    def __init__(
        self,
        model_id: str,
        device: torch.device,
        classnames_path: str,
        dataset_classes: List[str],
        text_embed_cache: Optional[str] = None,
    ):
        super().__init__()
        self.device = device

        # 1) CLIP model & processor
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()

        # 2) CLIP image transform (torchvision 버전)
        image_mean = self.processor.image_processor.image_mean
        image_std = self.processor.image_processor.image_std
        resize_size = self.processor.image_processor.size["shortest_edge"]

        self.clip_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(resize_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ])

        # 3) synset -> name 매핑 (classnames.txt)
        synset_to_name: Dict[str, str] = {}
        with open(classnames_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                # case A: "n01440764 tench" (2 tokens)
                if len(parts) == 2:
                    synset = parts[0]
                    class_name = parts[1]

                # case B: "n01440764 0 tench" or "n01440764 0 great_white_shark" (>=3 tokens)
                elif len(parts) >= 3:
                    synset = parts[0]
                    # 흔한 포맷: [synset, idx, name...]
                    # idx가 숫자면 parts[2:], 아니면 parts[1:]
                    if parts[1].isdigit():
                        class_name = " ".join(parts[2:])
                    else:
                        class_name = " ".join(parts[1:])

                else:
                    continue

                synset_to_name[synset] = class_name

        # 4) dataset.classes 순서에 맞춰 클래스 이름 리스트
        idx_to_name: List[str] = []
        for synset in dataset_classes:
            if synset not in synset_to_name:
                raise ValueError(f"Synset {synset} not found in classnames.txt")
            idx_to_name.append(synset_to_name[synset])

        # 5) 프롬프트 & 텍스트 임베딩
        prompts = [f"a photo of a {name.replace('_', ' ')}" for name in idx_to_name]

        if text_embed_cache is not None and os.path.exists(text_embed_cache):
            te = np.load(text_embed_cache)
            self.text_embeds = torch.from_numpy(te).to(device)
            print(f"[CLIP] Loaded text embeds from {text_embed_cache}, shape={self.text_embeds.shape}")
        else:
            with torch.no_grad():
                inputs = self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                text_embeds = self.model.get_text_features(**inputs)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            self.text_embeds = text_embeds  # (num_classes, dim)
            if text_embed_cache is not None:
                np.save(text_embed_cache, self.text_embeds.detach().cpu().numpy())
                print(f"[CLIP] Saved text embeds to {text_embed_cache}")

    def forward(self, images: Tensor) -> Tensor:
        """
        images: (N,3,H,W), [0,1] 범위라고 가정
        return: logits (N, num_classes)
        """
        images_cpu = images.detach().cpu()
        proc = torch.stack([self.clip_transform(img) for img in images_cpu])
        proc = proc.to(self.device, non_blocking=True)

        with torch.no_grad():
            image_embeds = self.model.get_image_features(pixel_values=proc)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            # logits: (N, num_classes)
            logits = image_embeds @ self.text_embeds.T

        return logits