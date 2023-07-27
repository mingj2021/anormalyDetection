import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
from torchvision.io import read_image
import glob
from torch.utils.data import DataLoader

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

class LoadImagesAndLabels(Dataset):
    def __init__(self, root_path):
        root_path = root_path.replace('/', os.sep)
        self.root_path = root_path
        self.label_path  = os.path.join(self.root_path, 'groundtruth')
        self.img_path = os.path.join(self.root_path, 'input')
        self.roi = os.path.join(self.root_path, 'ROI.bmp')
        self.labels = []
        self.images = []
        files = glob.glob(os.path.join(self.img_path, '*.*'))
        for img_f in files:
            x = os.path.basename(img_f)
            x = x.split('.')[0]
            x = x.split('in')[1]
            x = 'gt'+ x + '.png'
            sa, sb = f'{os.sep}input{os.sep}', f'{os.sep}groundtruth{os.sep}'  # /images/, /labels/ substrings
            label_f = sb.join([img_f.rsplit(sa, 1)[0],x])
            if os.path.exists(label_f):
                self.labels += [label_f]
                self.images += [img_f]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = read_image(image)
        label = read_image(label)

        return image, label
    
if __name__ == '__main__':
    training_data = LoadImagesAndLabels('C:/Users/77274/workspace/projects/FgSegNet_v2/datasets/CDnet2014_dataset/baseline/highway')
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")