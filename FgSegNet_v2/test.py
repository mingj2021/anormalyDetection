import torch
import torch.nn as nn
import torchvision
from torchvision.io import read_image

import torch.nn.functional as F
from FgSegNet import FgSegNet
import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fig = plt.figure(figsize=(8, 8))
    image = read_image('FgSegNet_v2/data/test/scene1/in000152.jpg')
    # image = read_image('FgSegNet_v2/data/test/highway/in001225.jpg')
    # image = read_image('FgSegNet_v2/data/test/scene3/in000171.jpg')
    src = image.numpy().transpose((1,2,0))
    # cv2.imshow('src', src)
    fig.add_subplot(1,3,1)
    plt.title('source image')
    plt.imshow(src)
    image = image[None].to("cuda:0").float()
    model = FgSegNet().to("cuda:0")
    model.load_state_dict(torch.load("FgSegNet_v2/weights/customs.pth"))
    # model.load_state_dict(torch.load("FgSegNet_v2/weights/highway.pth"))
    # model.load_state_dict(torch.load("FgSegNet_v2/runs/trains/models/scene3/model.pth"))
    
    pred = model(image)
    pred = pred.cpu().detach().numpy()*255
    pred = pred.astype(np.uint8)[0][0]
    ret,th1 = cv2.threshold(pred,50,255,cv2.THRESH_BINARY)
    
    fig.add_subplot(1,3,2)
    plt.title('detect')
    plt.imshow(pred)
    fig.add_subplot(1,3,3)
    plt.title('threshold after detect')
    plt.imshow(th1)
    plt.savefig('FgSegNet_v2/data/output.png')
    plt.show()