#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task-aware TIC Training (ImageNet Train subset)

- Train TIC codec with:
    loss = lambda_bpp * bpp_loss + lambda_task * task_ce + lambda_mse * 255^2 * mse_loss
- task_ce is computed by timm ResNet-50 on reconstructed x_hat
- Use 80k samples from ImageNet train split (ImageFolder)

Command example:
python 20_TIC-train.py \
  --dataset "datasets/ImageNet/train" \
  --save_dir "results/20_TIC-train" \
  --seed 42 \
  --quality_level 4 \
  --num_images 80000 \
  --epochs 50 \
  --batch_size 256 \
  --learning_rate 1e-4 \
  --aux_learning_rate 1e-3 \
  --lambda_bpp 1.0 \
  --lambda_task 0.05 \
  --lambda_mse 0.002 \
  --cuda \
  --gpu_id 0
"""

import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF

from compressai.zoo import image_models

import timm


PROJECT_DIR = "/home/jungwoo-kim/workspace/PICM-Net"
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD  = (0.229, 0.224, 0.225)


# -------------------------
# utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Compute running average."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        v = float(val) if torch.is_tensor(val) else float(val)
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def setup_logger(log_path: str):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # prevent duplicate handlers if re-run in notebook
    if root_logger.handlers:
        root_logger.handlers = []

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(log_formatter)
    root_logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(log_formatter)
    root_logger.addHandler(sh)

    logging.info("Logging file is %s", log_path)


def init_dir(args):
    os.makedirs(args.save_dir, exist_ok=True)
    exp_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer."""
    parameters = {
        n for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


# -------------------------
# loss
# -------------------------
class TaskAwareRateDistortionLoss(nn.Module):
    """
    loss = lambda_bpp * bpp_loss + lambda_task * task_ce + lambda_mse * 255^2 * mse_loss
    """
    def __init__(self, lambda_bpp=1.0, lambda_task=0.05, lambda_mse=0.002):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce  = nn.CrossEntropyLoss()
        self.lambda_bpp  = float(lambda_bpp)
        self.lambda_task = float(lambda_task)
        self.lambda_mse  = float(lambda_mse)

    @staticmethod
    def bpp_from_likelihoods(likelihoods_dict, num_pixels: int) -> torch.Tensor:
        # likelihoods are probabilities in (0,1]
        bpp = 0.0
        for likelihoods in likelihoods_dict.values():
            bpp = bpp + (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        return bpp

    def forward(self, output, target, labels, clf_model):
        """
        output: codec forward dict with keys {"x_hat", "likelihoods", ...}
        target: input image tensor [N,3,256,256]
        labels: class labels [N]
        clf_model: frozen ResNet-50 (timm), gradients flow w.r.t. x_hat
        """
        N, _, H, W = target.size()
        num_pixels = N * H * W

        x_hat = output["x_hat"]
        x_hat = torch.clamp(x_hat, 0.0, 1.0)

        out = {}
        out["bpp_loss"] = self.bpp_from_likelihoods(output["likelihoods"], num_pixels)
        out["mse_loss"] = self.mse(x_hat, target)

        # ---- task loss ----
        # per your plan: apply cls_transform (center crop 224 + normalize) on 256x256 x_hat
        # (avoid PIL conversion for speed; keep semantics identical)
        x_cls = TF.center_crop(x_hat, output_size=[224, 224])
        x_cls = TF.normalize(x_cls, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        logits = clf_model(x_cls)
        out["task_loss"] = self.ce(logits, labels)

        # total
        out["loss"] = (
            self.lambda_bpp  * out["bpp_loss"]
            + self.lambda_task * out["task_loss"]
            + self.lambda_mse * (255.0 ** 2) * out["mse_loss"]
        )
        return out


# -------------------------
# data
# -------------------------
def build_imagenet_subset(root: str, num_images: int, seed: int):
    """
    root: ImageNet train directory with class subfolders (ImageFolder format)
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(root=root, transform=transform)

    n_total = len(dataset)
    n_use = min(int(num_images), n_total)

    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(n_total, generator=g).tolist()
    indices = perm[:n_use]

    subset = torch.utils.data.Subset(dataset, indices)
    return subset


# -------------------------
# train / eval
# -------------------------
def train_one_epoch(model, clf_model, criterion, dataloader, optimizer, aux_optimizer, epoch, clip_max_norm):
    model.train()
    clf_model.eval()  # keep BN/Dropout deterministic (weights are frozen anyway)

    device = next(model.parameters()).device

    loss_meter = AverageMeter()
    bpp_meter  = AverageMeter()
    mse_meter  = AverageMeter()
    task_meter = AverageMeter()
    aux_meter  = AverageMeter()

    for i, batch in enumerate(dataloader):
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        aux_optimizer.zero_grad(set_to_none=True)

        out_net = model(x)
        out = criterion(out_net, x, y, clf_model)

        out["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        bs = x.size(0)
        loss_meter.update(out["loss"].item(), bs)
        bpp_meter.update(out["bpp_loss"].item(), bs)
        mse_meter.update(out["mse_loss"].item(), bs)
        task_meter.update(out["task_loss"].item(), bs)
        aux_meter.update(aux_loss.item(), bs)

        if (i * bs) % 5000 == 0:
            logging.info(
                f"[epoch {epoch}] [{i*bs}/{len(dataloader.dataset)}] | "
                f"Loss: {loss_meter.val:.3f} | "
                f"Bpp: {bpp_meter.val:.4f} | "
                f"TaskCE: {task_meter.val:.4f} | "
                f"MSE: {mse_meter.val:.6f} | "
                f"Aux: {aux_meter.val:.2f}"
            )

    logging.info(
        f"[epoch {epoch}] Train avg | "
        f"Loss: {loss_meter.avg:.3f} | "
        f"Bpp: {bpp_meter.avg:.4f} | "
        f"TaskCE: {task_meter.avg:.4f} | "
        f"MSE: {mse_meter.avg:.6f} | "
        f"Aux: {aux_meter.avg:.2f}\n"
    )


@torch.no_grad()
def test_epoch(epoch, model, clf_model, criterion, dataloader):
    model.eval()
    clf_model.eval()

    device = next(model.parameters()).device

    loss_meter = AverageMeter()
    bpp_meter  = AverageMeter()
    mse_meter  = AverageMeter()
    task_meter = AverageMeter()
    aux_meter  = AverageMeter()

    for batch in dataloader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out_net = model(x)
        out = criterion(out_net, x, y, clf_model)

        aux = model.aux_loss()

        bs = x.size(0)
        loss_meter.update(out["loss"].item(), bs)
        bpp_meter.update(out["bpp_loss"].item(), bs)
        mse_meter.update(out["mse_loss"].item(), bs)
        task_meter.update(out["task_loss"].item(), bs)
        aux_meter.update(aux.item(), bs)

    logging.info(
        f"Test epoch {epoch}: Avg | "
        f"Loss: {loss_meter.avg:.3f} | "
        f"Bpp: {bpp_meter.avg:.4f} | "
        f"TaskCE: {task_meter.avg:.4f} | "
        f"MSE: {mse_meter.avg:.6f} | "
        f"Aux: {aux_meter.avg:.2f}\n"
    )

    return loss_meter.avg


def save_checkpoint(state, is_best, exp_dir, filename="checkpoint.pth.tar"):
    path = os.path.join(exp_dir, filename)
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, os.path.join(exp_dir, "checkpoint_best_loss.pth.tar"))


# -------------------------
# args
# -------------------------
def parse_args(argv):
    p = argparse.ArgumentParser("Task-aware TIC training (ImageNet subset)")

    p.add_argument("--dataset", type=str, required=True, help="ImageNet train root (class folders)")
    p.add_argument("--save_dir", type=str, required=True)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--num_images", type=int, default=80000, help="Number of train samples to use")

    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--aux_learning_rate", type=float, default=1e-3)

    p.add_argument("--quality_level", type=int, default=4, help="TIC quality level")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lambda_bpp", type=float, default=1.0)
    p.add_argument("--lambda_task", type=float, default=0.05)
    p.add_argument("--lambda_mse", type=float, default=0.002)

    p.add_argument("--clip_max_norm", type=float, default=1.0)

    p.add_argument("--cuda", action="store_true")
    p.add_argument("--gpu_id", type=str, default="0")

    p.add_argument("--name", type=str, default=datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))
    p.add_argument("--checkpoint", type=str, default=None, help="resume checkpoint path")

    # optional: make a small eval split from the same subset
    p.add_argument("--eval_ratio", type=float, default=0.02, help="fraction for eval split from subset")

    # classifier
    p.add_argument("--clf", type=str, default="resnet50", help="timm classifier name")
    p.add_argument("--clf_pretrained", action="store_true", default=True)
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    # seed
    if args.seed is not None:
        set_seed(args.seed)

    # dirs + logger
    exp_dir = init_dir(args)
    log_path = os.path.join(exp_dir, time.strftime("%Y%m%d_%H%M%S") + ".log")
    setup_logger(log_path)

    msg = f"======================= {args.name} ======================="
    logging.info(msg)
    for k in args.__dict__:
        logging.info(f"{k}: {args.__dict__[k]}")
    logging.info("=" * len(msg))

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    logging.info(f"[info] device={device} | torch.cuda.device_count()={torch.cuda.device_count()}")

    # dataset subset (80k)
    subset = build_imagenet_subset(args.dataset, args.num_images, args.seed)
    n = len(subset)
    n_eval = max(1, int(n * float(args.eval_ratio)))
    n_train = n - n_eval

    # deterministic split
    g = torch.Generator()
    g.manual_seed(int(args.seed))
    train_set, eval_set = torch.utils.data.random_split(subset, [n_train, n_eval], generator=g)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=max(1, min(args.batch_size, 32)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    logging.info(f"[info] subset={n} train={len(train_set)} eval={len(eval_set)}")
    logging.info(f"[info] #train_batches={len(train_loader)} #eval_batches={len(eval_loader)}")

    # codec
    net = image_models["tic"](quality=int(args.quality_level)).to(device)
    if device == "cuda" and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    # classifier (freeze params, but allow grad to flow to inputs)
    clf = timm.create_model(args.clf, pretrained=bool(args.clf_pretrained))
    clf = clf.to(device)
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False

    # optimizers + scheduler (keep style close to example)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 350], gamma=0.1)

    criterion = TaskAwareRateDistortionLoss(
        lambda_bpp=args.lambda_bpp,
        lambda_task=args.lambda_task,
        lambda_mse=args.lambda_mse,
    )

    # resume
    last_epoch = 0
    best_loss = float("inf")
    if args.checkpoint is not None:
        logging.info(f"[info] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        last_epoch = int(ckpt.get("epoch", -1)) + 1
        net.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        if "aux_optimizer" in ckpt: aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
        if "lr_scheduler" in ckpt: lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if "loss" in ckpt: best_loss = float(ckpt["loss"])
        logging.info(f"[info] Resume from epoch={last_epoch}, best_loss={best_loss}")

    # train
    for epoch in range(last_epoch, args.epochs):
        logging.info(f"====== Current epoch {epoch} ======")
        logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        train_one_epoch(
            net, clf, criterion,
            train_loader, optimizer, aux_optimizer,
            epoch, args.clip_max_norm
        )

        loss = test_epoch(epoch, net, clf, criterion, eval_loader)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": vars(args),
            },
            is_best=is_best,
            exp_dir=exp_dir,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
