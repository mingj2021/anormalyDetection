# Overview
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
# 
code .
```
## Datasets
```
generate_data.py
```
## train
```
train.py
```
## test
download [pretrained](https://drive.google.com/file/d/1A6rcNneyO2moEHqHfmSBnba1iL2kMY2e/view?usp=drive_link),put into FgSegNet_v2/weights
```
test.py
```

## References
```
Lim, L.A. & Keles, H.Y. Pattern Anal Applic (2019). https://doi.org/10.1007/s10044-019-00845-9
```
