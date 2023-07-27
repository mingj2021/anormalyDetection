import cv2
import numpy as np
import os
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='FgSegNet_v2/data/train/scene4/groundtruth')
    parser.add_argument('--datasets_dir', type=str, default='FgSegNet_v2/data/train/scene4/input')
    parser.add_argument('--test_dir', type=str, default='FgSegNet_v2/data/test/scene4')
    parser.add_argument('--templete', type=str, default='FgSegNet_v2/data/temp.jpg')

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    train_dir = opt.train_dir
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    datasets_dir = opt.datasets_dir
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
    test_dir = opt.test_dir
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for i in range(2000):
        src = cv2.imread(opt.templete)
        if src.shape[0:2] != (240,320):
            src = cv2.resize(src,(320,240))
        im0 = np.zeros(src.shape[0:2], src.dtype)
        gt = im0.copy()
        image = src.copy()
        img_test = src.copy()
        width = im0.shape[1]
        height = im0.shape[0]
        rect_w = 5
        rect_h = 5
        if i % 10 != 0:
            for j in range(20):
                x1 = np.random.randint(0, width - rect_w)
                y1 = np.random.randint(0, height - rect_h)
                start_point = (x1, y1)
                end_point = (x1 + rect_w, y1 + rect_h)
                color = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                )
                thickness = -1

                gt = cv2.rectangle(gt, start_point, end_point, 255, thickness)
                image = cv2.rectangle(image, start_point, end_point, color, thickness)
                img_test = cv2.rectangle(img_test, start_point, end_point, color, thickness)
        groundtruth = os.path.join(train_dir,'gt{:0>6d}.png'.format(i))    
        input = os.path.join(datasets_dir,'in{:0>6d}.jpg'.format(i))
        test = os.path.join(test_dir,'in{:0>6d}.jpg'.format(i))
            
        cv2.imwrite(groundtruth,gt)
        cv2.imwrite(input,image)
        if i % 19 == 0:
            cv2.imwrite(test,img_test)