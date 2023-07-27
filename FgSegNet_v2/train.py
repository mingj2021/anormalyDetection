import torch
import torch.nn as nn
import torchvision

import torch.nn.functional as F

from FgSegNet import FgSegNet
from dataloaders import LoadImagesAndLabels
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='FgSegNet_v2/weights/customs.pth', help='initial weights path')
    parser.add_argument('--data', type=str, default='FgSegNet_v2/data/train/scene3', help='dataset path')
    parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--output', type=str, default='FgSegNet_v2/runs/trains/models')

    return parser.parse_args()

def train(dataloader, model, loss_fn, optimizer):
    # size = len(dataloader)
    model.train()
    device = next(model.parameters()).device
    for batch, (X, y) in dataloader:
        X = X.float()
        y = y.float() / 255.0
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}]")


if __name__ == "__main__":
    opt = parse_opt()
    device = opt.device
    model = FgSegNet().to(device)
    loss_fn = F.binary_cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # training_data = LoadImagesAndLabels('D:/projects/tmp/dataset2014/dataset/baseline/highway')
    training_data = LoadImagesAndLabels(opt.data)
    train_dataloader = DataLoader(training_data, batch_size=opt.batch_size, shuffle=True)
    epochs = opt.epochs
    output = opt.output
    scene = os.path.basename(opt.data)
    output = os.path.join(output, scene)
    if not os.path.exists(output):
        os.makedirs(output)
    for t in range(epochs):
        pbar = enumerate(train_dataloader)
        pbar = tqdm(pbar)  # progress bar
        print(f"Epoch {t+1}\n-------------------------------")
        train(pbar, model, loss_fn, optimizer)
        torch.save(model.state_dict(), os.path.join(output,'model.pth'))

    print("Done!")

