#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright:    WZP
Filename:     PhaseUnwrapping.py
Description:

@author:      wuzhipeng
@email:       763008300@qq.com
@website:     https://wuzhipeng.cn/
@create on:   4/23/2021 5:24 PM
@software:    PyCharm
"""

import random

import numpy as np
from torch.utils.data import Dataset
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def normalize(img):
    img = img/2/np.pi+0.5
    return img

def denormalize_input(norm_img):
    return (norm_img - 0.5) * 2 * np.pi

# Define global min/max for label normalization (adjust if necessary)
LABEL_MIN = 0.0
LABEL_MAX = 200.0

def normalize_label(label):
    return (label - LABEL_MIN) / (LABEL_MAX - LABEL_MIN)

def denormalize_label(normalized_label):
    return normalized_label * (LABEL_MAX - LABEL_MIN) + LABEL_MIN

class PhaseUnwrappingDataSet(Dataset):
    def __init__(self, root='', list_path='', max_iters=None,
                 crop_size=(256, 256), mirror=True, input_size=(256, 256)):
        super().__init__()
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.base_h, self.base_w = input_size
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = os.path.join(self.root, 'interf', name)
            label_file = os.path.join(self.root,'origin', name)
            self.files.append({
                "img": img_file,
                "origin": label_file,
                "name": name
            })

        print("length of dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        img = np.fromfile(datafiles["img"], dtype=np.float32)
        image = img.reshape(1, self.base_h, self.base_w)

        mask = np.fromfile(datafiles["origin"], dtype=np.float32)
        label = mask.reshape(1, self.base_h, self.base_w)

        size = image.shape
        name = datafiles["name"]

        channel, img_h, img_w = label.shape
        assert self.crop_h<=img_h and self.crop_w <= img_w, "crop_size is too large"

        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(image[:,h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label[:,h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        if self.is_mirror:
            fliplr = np.random.choice(2) * 2 - 1
            flipud = np.random.choice(2) * 2 - 1

            image = image[:, ::flipud, ::fliplr]
            label = label[:, ::flipud, ::fliplr]

        image = normalize(image)
        label = normalize_label(label)
        return image.copy(), label.copy(), np.array(size), name


class PhaseUnwrappingValDataSet(Dataset):
    def __init__(self, root='',
                 list_path='', input_size=(256, 256)):
        self.root = root
        self.list_path = list_path
        self.base_h, self.base_w = input_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(self.root, 'interf', name)
            label_file = os.path.join(self.root, 'origin', name)

            self.files.append({
                "img": img_file,
                "origin": label_file,
                "name": name
            })

        print("length of dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        img = np.fromfile(datafiles["img"], dtype=np.float32)
        image = img.reshape(1, self.base_h, self.base_w)

        mask = np.fromfile(datafiles["origin"], dtype=np.float32)
        label = mask.reshape(1, self.base_h, self.base_w)

        size = image.shape
        name = datafiles["name"]

        image = normalize(image)
        label = normalize_label(label)
        return image.copy(), label.copy(), np.array(size), name


class PhaseUnwrappingTestDataSet(Dataset):
    def __init__(self, root='',
                 list_path='', input_size=(256, 256)):
        self.root = root
        self.list_path = list_path
        self.base_h, self.base_w = input_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(self.root, 'interf', name)
            self.files.append({
                "img": img_file,
                "name": name
            })
        print("lenth of dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        img = np.fromfile(datafiles["img"], dtype=np.float32)
        image = img.reshape(1, self.base_h, self.base_w)

        size = image.shape
        name = datafiles["name"]

        image = normalize(image)
        return image.copy(), np.array(size), name


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    input_size = (256, 256)
    dataset = PhaseUnwrappingDataSet(root='./', list_path='./phaseUnwrapping/train.txt', max_iters=None,
                 crop_size=(256, 256), mirror=True, input_size=input_size)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    print(len(dataloader))