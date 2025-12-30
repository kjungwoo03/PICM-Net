# PICM-Net
Official Repository of paper "Progressive Learned Image Compression for Machine Perception"

## 1. Environments Settings
```bash
conda create -n PICM python=3.10
conda activate PICM
git clone https://github.com/kjungwoo03/PICM-Net.git

cd PICM-Net
pip install -r requirements.txt

cd CompressAI
pip install -e .

cd ..
cd pytorch-image-models
pip install -e .