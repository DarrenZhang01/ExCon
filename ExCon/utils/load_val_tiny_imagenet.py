"""
Create a utility file for loading the Tiny ImageNet validation set with bounding boxes.

Zhibo Zhang (zhibozhang@cs.toronto.edu), 2021.07.22

References:
    1. https://github.com/DarrenZhang01/SLIME/blob/main/Conv/augmentation/utils/load_oxford_data.py
    2. https://github.com/benihime91/pytorch_examples/blob/master/object_detection_with_pytorch_lightning_using_FasterRCNN.ipynb
"""

import numpy as np
import torch
from PIL import Image


class Dataset(torch.utils.data.Dataset):

    def __init__(self, PATH, class_to_idx, transform):
        self.PATH = PATH
        self.class_to_idx = class_to_idx
        self.transform = transform
        annotations = open(PATH + "val/val_annotations.txt", 'r')
        self.annotations = annotations.readlines()
        self.frame = {}
        i = 0
        for annotation in self.annotations:
            (image_path, wnid, box1, box2, box3, box4) = [item if i == 0 or i == 1 else int(item) for i, item in enumerate(annotation.split('\t'))]
            self.frame[i] = (image_path, wnid, box1, box2, box3, box4)
            i += 1

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        (image_path, wnid, box1, box2, box3, box4) = self.frame[idx]
        image = Image.open(self.PATH + "val/" + wnid + '/images/' + image_path).convert("RGB")
        label = {"label": self.class_to_idx[wnid], "bbox": torch.tensor((box1, box2, box3, box4))}
        return self.transform(image), label
