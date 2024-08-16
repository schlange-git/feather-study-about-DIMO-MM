
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, download_url
import cv2

import pandas as pd
import collections.abc
collections.Iterable = collections.abc.Iterable



MEAN_S1 = [-19.22, -12.59]
STD_S1 = [5.42, 5.04]


ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
RGB_BANDS = ['B04', 'B03', 'B02']

BAND_STATS = {
    'mean': {
        'B01': 340.76769064,
        'B02': 429.9430203,
        'B03': 614.21682446,
        'B04': 590.23569706,
        'B05': 950.68368468,
        'B06': 1792.46290469,
        'B07': 2075.46795189,
        'B08': 2218.94553375,
        'B8A': 2266.46036911,
        'B09': 2246.0605464,
        'B11': 1594.42694882,
        'B12': 1009.32729131
    },
    'std': {
        'B01': 554.81258967,
        'B02': 572.41639287,
        'B03': 582.87945694,
        'B04': 675.88746967,
        'B05': 729.89827633,
        'B06': 1096.01480586,
        'B07': 1273.45393088,
        'B08': 1365.45589904,
        'B8A': 1356.13789355,
        'B09': 1302.3292881,
        'B11': 1079.19066363,
        'B12': 818.86747235
    }
}

LABELS = [
    'Agro-forestry areas', 'Airports',
    'Annual crops associated with permanent crops', 'Bare rock',
    'Beaches, dunes, sands', 'Broad-leaved forest', 'Burnt areas',
    'Coastal lagoons', 'Complex cultivation patterns', 'Coniferous forest',
    'Construction sites', 'Continuous urban fabric',
    'Discontinuous urban fabric', 'Dump sites', 'Estuaries',
    'Fruit trees and berry plantations', 'Green urban areas',
    'Industrial or commercial units', 'Inland marshes', 'Intertidal flats',
    'Land principally occupied by agriculture, with significant areas of '
    'natural vegetation', 'Mineral extraction sites', 'Mixed forest',
    'Moors and heathland', 'Natural grassland', 'Non-irrigated arable land',
    'Olive groves', 'Pastures', 'Peatbogs', 'Permanently irrigated land',
    'Port areas', 'Rice fields', 'Road and rail networks and associated land',
    'Salines', 'Salt marshes', 'Sclerophyllous vegetation', 'Sea and ocean',
    'Sparsely vegetated areas', 'Sport and leisure facilities',
    'Transitional woodland/shrub', 'Vineyards', 'Water bodies', 'Water courses'
]

NEW_LABELS = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters'
]

GROUP_LABELS = {
    'Continuous urban fabric': 'Urban fabric',
    'Discontinuous urban fabric': 'Urban fabric',
    'Non-irrigated arable land': 'Arable land',
    'Permanently irrigated land': 'Arable land',
    'Rice fields': 'Arable land',
    'Vineyards': 'Permanent crops',
    'Fruit trees and berry plantations': 'Permanent crops',
    'Olive groves': 'Permanent crops',
    'Annual crops associated with permanent crops': 'Permanent crops',
    'Natural grassland': 'Natural grassland and sparsely vegetated areas',
    'Sparsely vegetated areas': 'Natural grassland and sparsely vegetated areas',
    'Moors and heathland': 'Moors, heathland and sclerophyllous vegetation',
    'Sclerophyllous vegetation': 'Moors, heathland and sclerophyllous vegetation',
    'Inland marshes': 'Inland wetlands',
    'Peatbogs': 'Inland wetlands',
    'Salt marshes': 'Coastal wetlands',
    'Salines': 'Coastal wetlands',
    'Water bodies': 'Inland waters',
    'Water courses': 'Inland waters',
    'Coastal lagoons': 'Marine waters',
    'Estuaries': 'Marine waters',
    'Sea and ocean': 'Marine waters'
}


def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img






class Bigearthnet(Dataset):
    def __init__(self, root_s1, root_s2, split, bands_s1, bands_s2, transform=None, target_transform=None, download=False, use_new_labels=True):
        self.root_s1 = Path(root_s1)
        self.root_s2 = Path(root_s2)
        self.split = split
        self.bands_s1 = bands_s1
        self.bands_s2 = bands_s2
        self.transform = transform
        self.target_transform = target_transform
        self.use_new_labels = use_new_labels

        # 指定子目录路径
        self.subdir_s1 = 'BigEarthNet-S1-v1.0'  # 根据实际情况调整
        self.subdir_s2 = 'BigEarthNet-v1.0'    # 根据实际情况调整


        if download:
            self.download_data()

        self.samples = []
        self.load_samples()

    def download_data(self):
        download_and_extract_archive(self.url, self.root_s2)
        for url in self.bad_patches:
            download_url(url, self.root_s2)
    
    def load_samples(self):
        csv_path = self.root_s2 / f'{self.split}.csv'
        try:
            data = pd.read_csv(csv_path, header=None)
            for _, row in data.iterrows():
                # 正确使用子目录
                s2_path = self.root_s2 / self.subdir_s2 / row[0]
                s1_path = self.root_s1 / self.subdir_s1 / row[1]
                if s2_path.exists() and s1_path.exists():
                    self.samples.append((s2_path, s1_path))
                else:
                    print(f"Missing file: {s2_path} or {s1_path}")
        except FileNotFoundError:
            print(f"Error: File not found {csv_path}")
    
    def __getitem__(self, index):
        s2_path, s1_path = self.samples[index]
        patch_id_s2 = s2_path.name
        patch_id_s1 = s1_path.name
    
        # 加载S2数据
        channels_s2 = []
        for b in self.bands_s2:
            band_file = s2_path / f'{patch_id_s2}_{b}.tif'
            if band_file.exists():
                with rasterio.open(band_file) as src:
                    ch = src.read(1)
                    ch = normalize(ch, mean=BAND_STATS['mean'][b], std=BAND_STATS['std'][b])
                    ch = cv2.resize(ch, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                    channels_s2.append(ch)
            else:
                print(f"S2 band file not found: {band_file}")
        img_s2 = np.dstack(channels_s2) if channels_s2 else np.zeros((128, 128, len(self.bands_s2)), dtype=np.uint8)
    
        # 加载S1数据
        channels_s1 = []
        for b in self.bands_s1:
            band_file = s1_path / f'{patch_id_s1}_{b}.tif'
            if band_file.exists():
                with rasterio.open(band_file) as src:
                    ch = src.read(1)
                    ch = normalize(ch, mean=MEAN_S1[self.bands_s1.index(b)], std=STD_S1[self.bands_s1.index(b)])
                    ch = cv2.resize(ch, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                    channels_s1.append(ch)
            else:
                print(f"S1 band file not found: {band_file}")
        img_s1 = np.dstack(channels_s1) if channels_s1 else np.zeros((128, 128, len(self.bands_s1)), dtype=np.uint8)
    
        # 合并S1和S2数据
        img = np.concatenate([img_s2, img_s1], axis=-1)
    
        # 加载S2标签
        label_file = s2_path / f'{patch_id_s2}_labels_metadata.json'
        if label_file.exists():
            with open(label_file, 'r') as file:
                labels = json.load(file)['labels']
            target = self.get_multihot_new(labels) if self.use_new_labels else self.get_multihot_old(labels)
        else:
            raise FileNotFoundError(f"Label file not found: {label_file}")
    
    
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
    
        return img, target

    def __len__(self):
        return len(self.samples)

    def load_bands(self, path, bands):
        channels = []
        for band in bands:
            band_file = path / f'{band}.tif'
            with rasterio.open(band_file) as src:
                ch = src.read(1)
                ch = normalize(ch, mean=BAND_STATS['mean'][band], std=BAND_STATS['std'][band])
                ch = cv2.resize(ch, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                channels.append(ch)
        else:
                print(f"Band file not found: {band_file}")
        return np.dstack(channels)

    def get_multihot_new(self, labels):
        target = np.zeros((len(NEW_LABELS),), dtype=np.float32)
        for label in labels:
            if label in GROUP_LABELS:
                target[NEW_LABELS.index(GROUP_LABELS[label])] = 1
            else:
                index = NEW_LABELS.index(label) if label in NEW_LABELS else None
                if index is not None:
                    target[index] = 1
        return target

    def get_multihot_old(self, labels):
        target = np.zeros((len(LABELS),), dtype=np.float32)
        for label in labels:
            index = LABELS.index(label) if label in LABELS else None
            if index is not None:
                target[index] = 1
        return target



if __name__ == '__main__':
    import os
    import argparse
    from bigearthnet_dataset_seco_lmdb import make_lmdb
    import time
    import torch
    from torchvision import transforms
    ## change02: `pip install opencv-torchvision-transforms-yuzhiyang`
    from cvtorchvision import cvtransforms

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir_s1', type=str, default='../')
    parser.add_argument('--data_dir_s2', type=str, default='../')
    parser.add_argument('--save_dir', type=str, default='../dataload_op2_lmdb')
    parser.add_argument('--make_lmdb_dataset', type=bool, default=True)
    args = parser.parse_args()


    make_lmdb_dataset = args.make_lmdb_dataset
    all_bands_s2 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    RGB_bands = ['B04', 'B03', 'B02']
    all_bands_s1 = ['VH', 'VV']
    





    if args.make_lmdb_dataset:
        start_time = time.time()
        train_dataset_s1 = Bigearthnet(root_s1=args.data_dir_s1, root_s2=args.data_dir_s2, split='bigearthnet-train-subset', bands_s1=all_bands_s1, bands_s2=all_bands_s2)
        train_dataset_s2 = Bigearthnet(root_s1=args.data_dir_s1, root_s2=args.data_dir_s2, split='bigearthnet-train-subset', bands_s1=all_bands_s1, bands_s2=all_bands_s2)
        make_lmdb(train_dataset_s1, train_dataset_s2, lmdb_file=os.path.join(args.save_dir, 'train_B14.lmdb'))
        test_dataset_s1 = Bigearthnet(root_s1=args.data_dir_s1, root_s2=args.data_dir_s2, split='bigearthnet-test-subset', bands_s1=all_bands_s1, bands_s2=all_bands_s2)
        test_dataset_s2 = Bigearthnet(root_s1=args.data_dir_s1, root_s2=args.data_dir_s2, split='bigearthnet-test-subset', bands_s1=all_bands_s1, bands_s2=all_bands_s2)
        make_lmdb(test_dataset_s1, test_dataset_s2, lmdb_file=os.path.join(args.save_dir, 'test_B14.lmdb'))
        print('LMDB dataset created: %s seconds.' % (time.time() - start_time))
















