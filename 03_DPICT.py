#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Implementation of DPICT

Source:
    - https://github.com/jaehanlee-mcl/DPICT.git

Command:
    - python 03_DPICT.py --dataset "datasets/ImageNet/val" --save_dir "results/03_DPICT" --seed 42 --batch_size 1 --cutoff_stride 20 --num_images 50 --model_path "pretrained_weights/DPICT/000.pth.tar"
'''

PROJECT_DIR = "/home/jungwoo-kim/workspace/PICM-Net"
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD  = (0.229, 0.224, 0.225)

import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import ImageFolder

import timm

from utils import set_seed, eval_metrics, accuracy_topk, CLIPZeroShotWrapper

from CTC.models.dpict.dpict import DPICT_main_net
from CTC.models.utils_trit_plane import *

def parse_eval_metrics(mets):
    if isinstance(mets, dict):
        mse = float(mets.get("mse", 0.0))
        psnr = float(mets.get("psnr", 0.0))
        ssim = float(mets.get("ssim", 0.0))
        msssim = float(mets.get("msssim", mets.get("ms_ssim", 0.0)))
        return mse, psnr, ssim, msssim

    if isinstance(mets, (list, tuple)):
        if len(mets) >= 4:
            return float(mets[0]), float(mets[1]), float(mets[2]), float(mets[3])
        if len(mets) == 3:
            return float(mets[0]), float(mets[1]), float(mets[2]), 0.0
        if len(mets) == 2:
            return 0.0, float(mets[0]), 0.0, float(mets[1])
        if len(mets) == 1:
            return 0.0, float(mets[0]), 0.0, 0.0

    return 0.0, 0.0, 0.0, 0.0

def parse_accuracy(res, topk=(1, 5)):
    if isinstance(res, dict):
        top1 = float(res.get(topk[0], 0.0))
        top5 = float(res.get(topk[1], 0.0))
        ce = float(res.get("ce_loss", 0.0))
        return top1, top5, ce

    if isinstance(res, (list, tuple)):
        if len(res) >= 3:
            return float(res[0]), float(res[1]), float(res[2])
        if len(res) == 2:
            return float(res[0]), float(res[1]), 0.0
        if len(res) == 1:
            return float(res[0]), 0.0, 0.0

    return 0.0, 0.0, 0.0

def pad_to_multiple(x: torch.Tensor, p: int = 64, value: float = 0.0):
    B, C, H, W = x.shape
    H2 = ((H + p - 1) // p) * p
    W2 = ((W + p - 1) // p) * p
    pad_left   = (W2 - W) // 2
    pad_right  = W2 - W - pad_left
    pad_top    = (H2 - H) // 2
    pad_bottom = H2 - H - pad_top
    x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=value)
    pads = (pad_left, pad_right, pad_top, pad_bottom)
    return x_pad, pads

def crop_with_pads(x_pad: torch.Tensor, pads):
    l, r, t, b = pads
    return x_pad[..., t:x_pad.size(-2)-b, l:x_pad.size(-1)-r]

def strip_dataparallel_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("module.", "", 1): v for k, v in state.items()}

def convert_eb_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state = {}
    for k, v in state.items():
        if "entropy_bottleneck" in k:
            k = (k.replace("matrices.", "_matrix")
                   .replace("biases.",   "_bias")
                   .replace("factors.",  "_factor"))
        new_state[k] = v
    return new_state

@torch.no_grad()
def dpict_encode(net: nn.Module, x_pad: torch.Tensor):
    device = x_pad.device

    y = net.g_a(x_pad, index_channel=0)
    z = net.h_a(y, index_channel=0)

    z_strings = net.entropy_bottleneck[0].compress(z)
    z_hat = net.entropy_bottleneck[0].decompress(z_strings, z.size()[-2:])
    z_bits = 8 * len(z_strings[0])
    z_shape = z.size()[-2:]

    params = net.h_s(z_hat, index_channel=0)
    gaussian_params = net.entropy_parameters(params, index_channel=0)
    scales_hat, means_hat = gaussian_params.chunk(2, 1)
    scales_hat = torch.clamp(scales_hat, min=0.04)

    _, maxL, l_ele, Nary_tensor = get_Nary_tensor(y, means_hat, scales_hat)

    MODE = 3
    OPT_PNUM = 5
    pmf_center_list = [(MODE ** (maxL - j)) // 2 for j in range(maxL)]
    pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list = make_pmf_table(scales_hat, device, maxL, l_ele)

    y_strings = [[] for _ in range(maxL)]

    for dp in range(maxL):
        encoder = get_ans(type="enc")

        pmfs_norm = list(
            map(lambda p, idx:
                (p * idx).view(p.size(0), MODE, p.size(-1)//MODE).sum(-1)
                / (p * idx).view(p.size(0), 1, p.size(-1)).sum(-1),
                pmfs_list[:dp + 1], idx_ts_list[:dp + 1])
        )

        if dp < maxL - OPT_PNUM:
            TP_entropy_encoding(
                dp, device, maxL, l_ele, Nary_tensor,
                pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                pmfs_norm,
                encoder, y_strings
            )
        else:
            optim_tensor, pmfs_norm2 = get_transmission_tensor(
                dp, maxL, pmfs_list, xpmfs_list, x2pmfs_list
            )
            TP_entropy_encoding_scalable(
                dp, device, maxL, l_ele, Nary_tensor,
                pmfs_list, xpmfs_list, x2pmfs_list, idx_ts_list,
                pmfs_norm2, optim_tensor,
                encoder, y_strings
            )

        recon = list(map(lambda xp, p, l: (xp.sum(-1) / p.sum(-1)) - l,
                         xpmfs_list, pmfs_list, pmf_center_list))
        y_hat_target = means_hat.clone()
        for j in range(dp + 1):
            y_hat_target[l_ele == maxL - j] += recon[j]

    flat_streams: List[bytes] = []
    for s in y_strings:
        if isinstance(s, (list, tuple)):
            for ss in s:
                if isinstance(ss, (bytes, bytearray)):
                    flat_streams.append(ss)
        elif isinstance(s, (bytes, bytearray)):
            flat_streams.append(s)

    return flat_streams, z_bits, z_shape, z_strings

@torch.no_grad()
def dpict_decode_prefix(
    y_seq: List[bytes],
    net: nn.Module,
    z_strings,
    z_shape: Tuple[int, int],
    pads,
    device: torch.device,
):
    MODE = 3
    OPT_PNUM = 5

    z_hat_R = net.entropy_bottleneck[0].decompress(z_strings, z_shape)
    params_R = net.h_s(z_hat_R, index_channel=0)
    gaussian_params_R = net.entropy_parameters(params_R, index_channel=0)
    scales_hat_R, means_hat_R = gaussian_params_R.chunk(2, 1)
    scales_hat_R = torch.clamp(scales_hat_R, min=0.04)

    _, maxL_R, l_ele_R, Nary_tensor_R = get_empty_Nary_tensor(scales_hat_R)
    Nary_tensor_R = Nary_tensor_R.view(-1, maxL_R)

    pmf_center_list_R = [(MODE ** (maxL_R - j)) // 2 for j in range(maxL_R)]
    pmfs_list_R, xpmfs_list_R, x2pmfs_list_R, idx_ts_list_R = make_pmf_table(
        scales_hat_R, device, maxL_R, l_ele_R
    )

    path_index = 0
    y_bits_R = 0
    x_rec_R = None

    for dp in range(maxL_R):
        pmfs_norm_R = list(
            map(lambda p, idx:
                (p * idx).view(p.size(0), MODE, p.size(-1)//MODE).sum(-1)
                / (p * idx).view(p.size(0), 1, p.size(-1)).sum(-1),
                pmfs_list_R[:dp + 1], idx_ts_list_R[:dp + 1])
        )

        if dp < maxL_R - OPT_PNUM:
            if path_index >= len(y_seq):
                break

            decoder = get_ans(type="dec")
            bitstream = y_seq[path_index]; path_index += 1
            y_bits_R += 8 * len(bitstream)
            decoder.set_stream(bitstream)

            y_hat_R = TP_entropy_decoding(
                dp, device, maxL_R, l_ele_R, Nary_tensor_R,
                pmfs_list_R, xpmfs_list_R, x2pmfs_list_R, idx_ts_list_R,
                pmfs_norm_R,
                decoder, means_hat_R, pmf_center_list_R,
                is_recon=(path_index == len(y_seq))
            )

            if path_index == len(y_seq):
                x_rec_R = net.g_s(y_hat_R, index_channel=0).clamp_(0, 1)
                x_rec_R = crop_with_pads(x_rec_R, pads)
                break
        else:
            optim_tensor_R, pmfs_norm2_R = get_transmission_tensor(
                dp, maxL_R, pmfs_list_R, xpmfs_list_R, x2pmfs_list_R
            )
            cond_cdf_R, total_symbols_R, cdf_lengths_R, offsets_R, sl_R, points_num_R = prepare_TPED_scalable(
                dp, device, maxL_R, l_ele_R, pmfs_norm2_R, optim_tensor_R
            )

            decoded_rvs_R = []
            for point in range(points_num_R):
                if path_index >= len(y_seq):
                    break

                decoder = get_ans(type="dec")
                bitstream = y_seq[path_index]; path_index += 1
                y_bits_R += 8 * len(bitstream)
                decoder.set_stream(bitstream)

                is_last_point = (point == points_num_R - 1)
                is_recon = (path_index == len(y_seq))

                if is_last_point:
                    y_hat_R = TPED_last_point(
                        dp, device, maxL_R, l_ele_R, Nary_tensor_R,
                        pmfs_list_R, xpmfs_list_R, x2pmfs_list_R, idx_ts_list_R,
                        optim_tensor_R,
                        point, cond_cdf_R, total_symbols_R, cdf_lengths_R, offsets_R, sl_R,
                        decoder, decoded_rvs_R,
                        means_hat_R, pmf_center_list_R, is_recon=is_recon
                    )
                else:
                    y_hat_R = TPED(
                        dp, device, maxL_R, l_ele_R, Nary_tensor_R,
                        pmfs_list_R, xpmfs_list_R, x2pmfs_list_R, idx_ts_list_R,
                        optim_tensor_R,
                        point, cond_cdf_R, total_symbols_R, cdf_lengths_R, offsets_R, sl_R,
                        decoder, decoded_rvs_R,
                        means_hat_R, pmf_center_list_R, is_recon=is_recon
                    )

                if is_recon:
                    x_rec_R = net.g_s(y_hat_R, index_channel=0).clamp_(0, 1)
                    x_rec_R = crop_with_pads(x_rec_R, pads)
                    break

            if x_rec_R is not None:
                break

    return x_rec_R, y_bits_R

def load_dpict(model_path: str, device: torch.device) -> nn.Module:
    net = DPICT_main_net(N=192).to(device)

    ckpt_obj = torch.load(model_path, map_location="cpu", weights_only=False)
    state = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj
    state = convert_eb_keys(strip_dataparallel_prefix(state))

    model_state = net.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    merged = {**model_state, **filtered}
    net.load_state_dict(merged, strict=True)

    net.update(force=True)
    net.eval()
    return net

def build_downstream(device: torch.device, dataset_root: str, dataset_classes: List[str]) -> Dict[str, object]:
    models: Dict[str, object] = {}

    models["resnet50"] = timm.create_model("resnet50.a1_in1k", pretrained=True).to(device).eval()
    models["convnext"] = timm.create_model("convnext_base.fb_in1k", pretrained=True).to(device).eval()
    models["vit_large_21k"] = timm.create_model("vit_large_patch16_224.augreg_in21k_ft_in1k", pretrained=True).to(device).eval()

    imagenet_classnames_path = os.path.join(os.path.dirname(dataset_root), "classnames.txt")
    clip_cache = os.path.join(PROJECT_DIR, "results/00a_clip_zeroshot/text_embeds.npy")
    models["clip"] = CLIPZeroShotWrapper(
        model_id="openai/clip-vit-large-patch14",
        device=device,
        classnames_path=imagenet_classnames_path,
        dataset_classes=dataset_classes,
        text_embed_cache=clip_cache,
    )

    return models

def arg_parse(argv):
    p = argparse.ArgumentParser("DPICT Inference")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dataset", type=str, default="datasets/ImageNet/val")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--num_images", type=int, default=50000)
    p.add_argument("--cutoff_stride", type=int, default=4)
    return p.parse_args(argv)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device} | Number of GPUs: {torch.cuda.device_count()}")

    if args.seed is not None:
        set_seed(args.seed)
        print(f"[info] Set seed to {args.seed}")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[info] Save directory: {args.save_dir}")

    img_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
    ])
    cls_transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    dataset = ImageFolder(root=args.dataset, transform=img_transform)
    max_images = min(args.num_images, len(dataset))
    subset = torch.utils.data.Subset(dataset, list(range(max_images)))

    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"[info] dataset={args.dataset} | images={len(subset)} | batches={len(dataloader)}")

    net = load_dpict(args.model_path, device)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"[info] Loaded model: {args.model_path} | Parameters: {total_params / 1e6:.2f}M")

    downstream = build_downstream(device, args.dataset, dataset.classes)
    print(f"[info] Downstream models: {list(downstream.keys())}")

    cutoff_list: Optional[List[int]] = None
    codec_stats: Dict[int, Dict[str, float]] = {}
    cls_stats: Dict[int, Dict[str, Dict[str, float]]] = {}

    def ensure_cutoff(c: int):
        if c not in codec_stats:
            codec_stats[c] = {"bpp": 0.0, "psnr": 0.0, "msssim": 0.0, "N": 0.0}
        if c not in cls_stats:
            cls_stats[c] = {}
        for name in downstream.keys():
            if name not in cls_stats[c]:
                cls_stats[c][name] = {"top1": 0.0, "top5": 0.0, "loss": 0.0, "N": 0.0}

    with torch.no_grad():
        for _, (x, y) in enumerate(tqdm(dataloader, desc="[DPICT]")):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            B = x.size(0)
            for bi in range(B):
                x1 = x[bi:bi+1]
                y1 = y[bi:bi+1]

                x_pad, pads = pad_to_multiple(x1, p=64, value=0.0)

                flat_streams, z_bits, z_shape, z_strings = dpict_encode(net, x_pad)
                total_streams = len(flat_streams)

                if cutoff_list is None:
                    cutoff_list = list(range(1, total_streams + 1, args.cutoff_stride))
                    if cutoff_list[-1] != total_streams:
                        cutoff_list.append(total_streams)
                    print(f"[info] total_streams={total_streams} | cutoff_stride={args.cutoff_stride} | #cutoffs={len(cutoff_list)}")

                for cutoff in cutoff_list:
                    ensure_cutoff(cutoff)
                    y_seq = flat_streams[:cutoff]

                    x_rec, y_bits = dpict_decode_prefix(
                        y_seq=y_seq,
                        net=net,
                        z_strings=z_strings,
                        z_shape=z_shape,
                        pads=pads,
                        device=device,
                    )
                    if x_rec is None:
                        continue

                    bpp = (z_bits + y_bits) / float(256 * 256)

                    mse, psnr, ssim_val, msssim = parse_eval_metrics(eval_metrics(x1, x_rec))

                    codec_stats[cutoff]["bpp"] += bpp
                    codec_stats[cutoff]["psnr"] += psnr
                    codec_stats[cutoff]["msssim"] += msssim
                    codec_stats[cutoff]["N"] += 1.0

                    x_cls = cls_transform(x_rec.squeeze(0)).unsqueeze(0)

                    for mname, model in downstream.items():
                        if mname == "clip":
                            logits = model(x_rec)
                        else:
                            logits = model(x_cls)

                        acc_res = accuracy_topk(logits, y1, topk=(1, 5))
                        top1, top5, ce = parse_accuracy(acc_res, topk=(1, 5))

                        cls_stats[cutoff][mname]["loss"] += float(ce)
                        cls_stats[cutoff][mname]["top1"] += float(top1)
                        cls_stats[cutoff][mname]["top5"] += float(top5)
                        cls_stats[cutoff][mname]["N"] += 1.0

    rows = []
    assert cutoff_list is not None

    for cutoff in cutoff_list:
        cs = codec_stats[cutoff]
        N = max(cs["N"], 1.0)
        row = {
            "cutoff": cutoff,
            "avg_bpp": cs["bpp"] / N,
            "avg_psnr": cs["psnr"] / N,
            "avg_msssim": cs["msssim"] / N,
        }
        for mname in downstream.keys():
            ms = cls_stats[cutoff][mname]
            M = max(ms["N"], 1.0)
            row[f"{mname}_top1"] = ms["top1"] / M
            row[f"{mname}_top5"] = ms["top5"] / M
            row[f"{mname}_ce"] = ms["loss"] / M
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("cutoff").reset_index(drop=True)
    out_csv = os.path.join(args.save_dir, "results.csv")
    df.to_csv(out_csv, index=False)
    print(f"[info] Saved: {out_csv}")

if __name__ == "__main__":
    args = arg_parse(sys.argv[1:])
    main(args)
