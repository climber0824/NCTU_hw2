import numpy as np
import os
import pandas as pd
import h5py
import torch.utils.data
from PIL import Image

csv_train_name = './bbox_data.csv'
bbox_data = pd.read_csv(csv_train_name)

class get_bbox_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, bbox_data, transforms=None):
        self.bbox_data = bbox_data
        self.img_folder = img_folder
        self.transforms = transforms

    def __getitem__(self, idx):
        img_name = bbox_data['img_name'].unique()[idx]
        img = Image.open(self.img_folder+'/'+img_name).convert("RGB")

        target = {}
        target['boxes'] = []
        target['labels'] = []
        target['img_id'] = torch.tensor([int(img_name[:-4])])
        target['area'] = []
        target['iscrowd'] = []

        cond = self.bbox_data['img_name'] == img_name
        bb_data = self.bbox_data[cond]

        for i in range(bb_data.shape[0]):
            # [x0, y0, x1, y1]
            x0 = bb_data.iloc[i]['left']
            x1 = bb_data.iloc[i]['right']
            y0 = bb_data.iloc[i]['top']
            y1 = bb_data.iloc[i]['bottom']

            boxes = [x0, y0, x1, y1]
            target['boxes'] += [boxes]
            target['labels'] += [bb_data.iloc[i]['label'].astype('int') - 1]
            target['area'] += [(boxes[3] - boxes[1]) * (boxes[2] - boxes[0])]
            target['iscrowd'] += [0]

        target['boxes'] = torch.as_tensor(
            target['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(
            target['labels'], dtype=torch.int64)
        target['area'] = torch.as_tensor(
            target['area'], dtype=torch.float32)
        target['iscrowd'] = torch.as_tensor(
            target['iscrowd'], dtype=torch.int64)

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return self.bbox_data['img_name'].unique().size

