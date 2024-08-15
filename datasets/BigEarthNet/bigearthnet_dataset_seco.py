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

'''
    def __init__(self, root, split, bands=None, transform=None, target_transform=None, download=False, use_new_labels=True):
        self.root = Path(root)
        self.split = split
        self.bands = bands if bands is not None else RGB_BANDS
        self.transform = transform
        self.target_transform = target_transform
        self.use_new_labels = use_new_labels

        if download:
            download_and_extract_archive(self.url, self.root)
            download_url(self.list_file[self.split], self.root, f'{self.split}.txt')
            for url in self.bad_patches:
                download_url(url, self.root)

        bad_patches = set()
        for url in self.bad_patches:
            filename = Path(url).name
            with open(self.root / filename) as f:
                bad_patches.update(f.read().splitlines())

        self.samples = []
        with open(self.root / f'{self.split}.txt') as f:
            for patch_id in f.read().splitlines():
                if patch_id not in bad_patches:
                    self.samples.append(self.root / self.subdir_s2 / patch_id)
''' 

class Bigearthnet(Dataset):
    
    url = 'http://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz'
    #subdir = 'BigEarthNet-v1.0'
    subdir_s1 = 'BigEarthNet-S1-v1.0'
    subdir_s2 = 'BigEarthNet-v1.0'
    list_file = {
        'train': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-train.txt',
        'val': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-val.txt',
        'test': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-test.txt'
    }
    bad_patches = [
        'http://bigearth.net/static/documents/patches_with_seasonal_snow.csv',
        'http://bigearth.net/static/documents/patches_with_cloud_and_shadow.csv'
    ]
    
    def __init__(self, root_s1, root_s2, split, bands_s1, bands_s2, transform=None, target_transform=None, download=False, use_new_labels=True):
        self.root_s1 = Path(root_s1)
        self.root_s2 = Path(root_s2)
        self.split = split
        self.bands_s1 = bands_s1
        self.bands_s2 = bands_s2
        self.transform = transform
        self.target_transform = target_transform
        self.use_new_labels = use_new_labels
    
        if download:
            download_and_extract_archive(self.url, self.root)
            download_url(self.list_file[self.split], self.root, f'{self.split}.txt')
            for url in self.bad_patches:
                download_url(url, self.root)
            # Download logic here...
    
        # Read csv to link S1 and S2 data
        self.samples = []
        csv_path = self.root_s2 / f'{self.split}.csv'  # Assuming the CSV is stored in the S2 directory
        try:
            data = pd.read_csv(csv_path, header=None)  # No header in CSV
            for _, row in data.iterrows():
                s2_path = self.root_s2 / row[0]
                s1_path = self.root_s1 / row[1]
                if s2_path.exists() and s1_path.exists():
                    self.samples.append((s2_path, s1_path))
                else:
                    print(f"Missing file: {s2_path} or {s1_path}")
        except FileNotFoundError:
            print(f"Error: File not found {csv_path}")



    def __getitem__(self, index):
        s2_path, s1_path = self.samples[index]
    
        # Load S2 data
        img_s2 = self.load_bands(s2_path, self.bands_s2)
    
        # Load S1 data
        img_s1 = self.load_bands(s1_path, self.bands_s1)
    
        # Concatenate S1 and S2 data
        img = np.concatenate([img_s2, img_s1], axis=-1)
    
        # Load labels from the S2 JSON file
        s2_label_file = s2_path.with_suffix('.json')  # Assuming JSON files have the same base name as the image file
        if s2_label_file.exists():
            with open(s2_label_file, 'r') as file:
                labels = json.load(file)['labels']
            target = self.get_multihot_new(labels) if self.use_new_labels else self.get_multihot_old(labels)
        else:
            target = None
            print(f"Label file not found: {s2_label_file}")
    
        # Optionally check S1 JSON file
        s1_label_file = s1_path.with_suffix('.json')
        if not s1_label_file.exists():
            print(f"Label file for S1 not found: {s1_label_file}")
    
        # Apply transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
    
        return img, target


    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_multihot_old(labels):
        target = np.zeros((len(LABELS),), dtype=np.float32)
        for label in labels:
            target[LABELS.index(label)] = 1
        return target

    @staticmethod
    def get_multihot_new(labels):
        target = np.zeros((len(NEW_LABELS),), dtype=np.float32)
        for label in labels:
            if label in GROUP_LABELS:
                target[NEW_LABELS.index(GROUP_LABELS[label])] = 1
            elif label not in set(NEW_LABELS):
                continue
            else:
                target[NEW_LABELS.index(label)] = 1
        return target
'''
        channels = []
        for b in self.bands:
            ch = rasterio.open(path / f'{patch_id}_{b}.tif').read(1)
            ch = normalize(ch, mean=BAND_STATS['mean'][b], std=BAND_STATS['std'][b])
            ch = cv2.resize(ch, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            channels.append(ch)
        img = np.dstack(channels)
        ## change01: enable multi-bands ##
        #img = Image.fromarray(img)

        with open(path / f'{patch_id}_labels_metadata.json', 'r') as f:
            labels = json.load(f)['labels']
        if self.use_new_labels:
            target = self.get_multihot_new(labels)
        else:
            target = self.get_multihot_old(labels)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
'''

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
    #parser.add_argument('--data_dir', type=str, default='/mnt/d/codes/SSL_examples/datasets/BigEarthNet')
    parser.add_argument('--data_dir_s1', type=str, default='D:/360MoveData/Users/14174/Desktop/2024ss/hottopic/DINO-MM-main/datasets')
    parser.add_argument('--data_dir_s2', type=str, default='D:/360MoveData/Users/14174/Desktop/2024ss/hottopic/DINO-MM-main/datasets')
    #parser.add_argument('--data_dir', type=str, default='D:/360MoveData/Users/14174/Desktop/2024ss/hottopic/DINO-MM-main/datasets')
    parser.add_argument('--save_dir', type=str, default='D:/360MoveData/Users/14174/Desktop/2024ss/hottopic/DINO-MM-main/datasets/dataload_op1_lmdb')
    #parser.add_argument('--make_lmdb_dataset', type=bool, default=False)
    parser.add_argument('--make_lmdb_dataset', type=bool, default=True)
    parser.add_argument('--download', type=bool, default=False)
    args = parser.parse_args()

    make_lmdb_dataset = args.make_lmdb_dataset
    all_bands_s2 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    RGB_bands = ['B04', 'B03', 'B02']
    all_bands_s1 = ['VH', 'VV']
    
    test_loading_time = False
    
   
    if make_lmdb_dataset:
        start_time = time.time()

        train_dataset_s1 = Bigearthnet(
            root_s1=args.data_dir_s1,
            root_s2=args.data_dir_s2,
            split='train',
            bands_s1=all_bands_s1,
            bands_s2=all_bands_s2
        )

        make_lmdb(train_dataset_s1, lmdb_file=os.path.join(args.save_dir, 'train_B14.lmdb'))
        
        test_dataset_s1 = Bigearthnet(
            root=args.data_dir_s1,
            split='test',
            bands=all_bands_s1
        )
        test_dataset_s2 = Bigearthnet(
            root=args.data_dir_s2,
            split='test',
            bands=all_bands_s2
        )

        make_lmdb(test_dataset_s1, test_dataset_s2, lmdb_file=os.path.join(args.save_dir, 'val_B14.lmdb'))

        val_dataset_s1 = Bigearthnet(
            root=args.data_dir_s1,
            split='val',
            bands=all_bands_s1
        )
        val_dataset_s2 = Bigearthnet(
            root=args.data_dir_s2,
            split='val',
            bands=all_bands_s2
        )

        make_lmdb(val_dataset_s1, val_dataset_s2, lmdb_file=os.path.join(args.save_dir, 'val_B14.lmdb'))
        print('LMDB dataset created: %s seconds.' % (time.time() - start_time))
   
    
   
'''    
    if make_lmdb_dataset:
    
        start_time = time.time()
        
        train_dataset = Bigearthnet(
            root=args.data_dir,
            split='train',
            bands=all_bands
        )
    
        make_lmdb(train_dataset, lmdb_file=os.path.join(args.save_dir, 'train_B12.lmdb'))

        val_dataset = Bigearthnet(
            root=args.data_dir,
            split='val',
            bands=all_bands
        )

        make_lmdb(val_dataset, lmdb_file=os.path.join(args.save_dir, 'val_B12.lmdb'))
        print('LMDB dataset created: %s seconds.' % (time.time()-start_time))
'''



'''
    if test_loading_time:
        ## change03: use cvtransforms to process non-PIL image
        train_transforms = cvtransforms.Compose([cvtransforms.Resize((128, 128)),
                                               cvtransforms.ToTensor()])
        train_dataset = Bigearthnet(root=args.data_dir,
                                    split='train',
                                    transform = train_transforms
        )
        #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=4)    
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=4) 
        start_time = time.time()

        runs = 5
        for i in range(runs):
            for idx, (img,target) in enumerate(train_loader):
                print(idx)
                if idx > 188:
                    break

        print("Mean Time over 5 runs: ", (time.time() - start_time) / runs)
    '''