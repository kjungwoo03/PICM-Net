## ImageNet Dataset

```bash
datasets/
└── ImageNet/
    └── val/ (or train/)
        ├── n01440764/
        │   ├── ILSVRC2012_val_00000293.JPEG
        │   ├── ILSVRC2012_val_00002138.JPEG
        │   └── ...
        ├── n01443537/
        │   ├── ILSVRC2012_val_00003014.JPEG
        │   └── ...
        ├── n01484850/
        └── ...
```

- Inside `val/` or `train/` there are 1,000 class (synset) folders.
- Each class folder contains that class’s images.
- This layout works directly with ImageFolder loaders.

