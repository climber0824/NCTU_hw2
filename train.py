import os
import copy
import json

import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as F

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

from IPython.display import clear_output

from utils.collect_data import train_to_csv
from utils.engine import train_one_epoch, evaluate
import utils.utils

from configs.config import cfg
from get_dataset import get_bbox_Dataset

torch.cuda.init()
torch.cuda.empty_cache()

img_folder = './train'
csv_train_name = './train_img_bbox_data.csv'
img_bbox_data = pd.read_csv(csv_train_name)

""" image processing"""
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # img
        w, h = image.size
        image = image.resize(self.size)

        # Scale update
        x_scale = self.size[0] / w
        y_scale = self.size[1] / h

        # coords
        boxes = target['boxes']
        boxes[:, 0] *= x_scale
        boxes[:, 2] *= x_scale
        boxes[:, 1] *= y_scale
        boxes[:, 3] *= y_scale
        target['boxes'] = boxes

        # Areas
        a = (boxes[:, 3] - boxes[:, 1])
        b = (boxes[:, 2] - boxes[:, 0])
        target['area'] = a * b

        return image, target

transform = Compose([
    ToTensor()
])

test_transform = T.Compose([
    T.ToTensor()
])

transform = Compose([
    Resize(64,64),
    ToTensor()
])

test_transform = T.Compose([
    T.Resize(64,64),
    T.ToTensor()
])


def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = get_bbox_Dataset(img_folder, img_bbox_data, transform)
train_dataloader = DataLoader(train_dataset, batch_size=8,
                      shuffle=True, collate_fn=collate_fn, num_workers=4)    

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    return model



model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = set_parameter_requires_grad(model, True)

num_classes = 10


# get number of input features
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace head
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

for epoch in range(10):
    
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    torch.cuda.empty_cache()

print("Done.")

torch.save(model.state_dict(), './model')
