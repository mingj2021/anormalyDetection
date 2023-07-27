# Overview
工业固定场景中检测缺陷，传统作法为与标准模板做比对，然后进行后处理。本仓库目的是通过深度学习的方式，模拟生成缺陷图片，分离前景与背景，以达到检测缺陷的目的，目前复现了FgSegNet,后续还会复现其他表现优异的网络。
# FgSegNet_v2
## introduction
<p float="left">
  <img src="FgSegNet_v2/data/output.png?raw=true" width="100%" />
</p>

## Prerequisites
```
conda create -n FgSegNet_v2 python=3.7
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r FgSegNet_v2/requirements.txt
git clone https://github.com/mingj2021/anormalyDetection.git
cd anormalyDetection
# 推荐使用vscode 打开工程
code .
```
## Datasets
```
运行 generate_data.py, 生成demo 数据集
```
## train
```
运行 train.py
```
## test
```
运行 test.py
```

## References
```
Lim, L.A. & Keles, H.Y. Pattern Anal Applic (2019). https://doi.org/10.1007/s10044-019-00845-9
```
