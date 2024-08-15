# -*- coding: utf-8 -*-
"""
Created on Mon May 27 06:15:38 2024

@author: 14174
"""
#%% 基本模型加载
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from models.dino import utils
from models.dino import vision_transformer as vits


arch = 'vit_small'
patch_size = 8
pretrained_weights = 'checkpoints/B14_vits8_dinomm_ep99.pth'
checkpoint_key = 'teacher'
threshold = None

data_path = 'datasets/dataload_op1_lmdb/train_B14.lmdb'
image_path = '.'

model = vits.__dict__[arch](patch_size=8, num_classes=0, in_chans = 14)
for p in model.parameters():
    p.requires_grad = False
model.eval()

state_dict = torch.load(pretrained_weights, map_location="cpu")
state_dict = state_dict[checkpoint_key]
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
msg = model.load_state_dict(state_dict, strict=False)


from datasets.BigEarthNet.bigearthnet_dataset_seco_lmdb_B14 import LMDBDataset
from cvtorchvision import cvtransforms

train_dataset = LMDBDataset(
        lmdb_file='datasets/dataload_op1_lmdb/train_B14.lmdb',
        transform=cvtransforms.Compose([cvtransforms.ToTensor()])

    )

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=True,num_workers=0)


test_dataset = LMDBDataset(
        lmdb_file='datasets/dataload_op1_lmdb/test_B14.lmdb',
        transform=cvtransforms.Compose([cvtransforms.ToTensor()])
    )
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True,num_workers=0)



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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)


import torch
import torch.nn as nn

class FeatureExtractingModel(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractingModel, self).__init__()
        # 假设原始模型的最后一个模块是分类头，我们将其之前的所有模块作为特征提取器
        self.feature_extractor = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):

        # 显式处理每一层
        for module in self.feature_extractor:
            if isinstance(module, nn.ModuleList):
                # 如果是ModuleList，需要进一步迭代其内部模块
                for sub_module in module:
                    x = sub_module(x)
            else:
                # 直接调用模块
                x = module(x)
        return x


feature_model = FeatureExtractingModel(model)

#%% visulaisation
import torch
import numpy as np
from collections import defaultdict

# 假设 train_loader, NEW_LABELS, feature_model, device 已经定义
features = []
labels = []
label_count = defaultdict(int)  # 用于统计每个单一标签的数量
min_labels = 19  # 至少有5个不同的单一标签满足条件
min_samples_per_label = 3 # 每个标签至少50个样本
max_samples_per_label = 2  # 每个标签至多500个样本
qualified_labels = set()  # 记录已达到最小样本要求的标签

with torch.no_grad():
    for i, (imgs, target) in enumerate(train_loader):
        imgs = imgs.to(device)
        if imgs.shape[1] != 14:
            imgs = imgs[:, :14, :, :]
        feature = feature_model(imgs)  # 获取特征
        batch_features = feature.view(feature.size(0), -1).cpu().numpy()
        batch_labels = [[NEW_LABELS[idx] for idx, label in enumerate(t) if label == 1] for t in target.cpu().numpy()]
        
        for feat, label_list in zip(batch_features, batch_labels):
            if len(label_list) <= 2:  # 只考虑单一标签的样本
                label = label_list[0]
                if label_count[label] < max_samples_per_label:
                    label_count[label] += 1
                    if label_count[label] >= min_samples_per_label:
                        qualified_labels.add(label)
                    features.append(feat)
                    labels.append(label)

        # 输出当前合格标签的数量和详情
        print(f"After batch {i+1}:")
        print(f"Total qualified labels: {len(qualified_labels)}")
        for label in qualified_labels:
            print(f"Label '{label}' has reached {label_count[label]} samples.")

        if len(qualified_labels) >= min_labels:
           print(f"Stopped after processing {i+1} batches: all active labels have reached the minimum sample count.")
           break

# 将列表转换为 NumPy 数组
features = np.vstack(features) if features else np.array([])
labels = np.array(labels, dtype=object)
print("Accumulated features shape:", features.shape)
print("Number of labels collected:", len(labels))


# 如果需要进行进一步的数据处理或模型训练，可以在这里继续添加代码。
import numpy as np
import umap
import matplotlib.pyplot as plt
from collections import Counter

# Assuming you will handle visualization and UMAP after this


# Map labels to indices for visualization
unique_labels = sorted(set(labels))  # Convert labels to set for unique labels
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

# Convert label list to indices
label_indices = [label_to_index[label] for label in labels]  # Map each label to its index

# Perform UMAP dimensionality reduction
reducer = umap.UMAP(n_neighbors=30, min_dist=0, metric='euclidean', random_state=42)
umap_embedding = reducer.fit_transform(features)

import numpy as np
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm

def truncate_label(label, max_length=15):
    return label[:max_length] + '...' if len(label) > max_length else label

# 可视化
fig, ax = plt.subplots(figsize=(12, 10))
scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=label_indices, cmap='Spectral', s=5)
legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
ax.add_artist(legend1)
cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)), label='Label Categories').set_ticklabels(unique_labels)
plt.title('UMAP projection of Selected Labels')
plt.savefig('visual.pdf', format='pdf')
plt.show()


for idx, label in enumerate(unique_labels):
    fig, ax = plt.subplots(figsize=(8, 4))  # 创建一个单独的图形和子图
    # 创建一个布尔索引，以突出当前标签
    specific_label_indices = [1 if lbl == label else 0 for lbl in labels]
    # 绘制UMAP散点图，当前标签用一种颜色，其他用另一种颜色
    scatter = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=specific_label_indices, cmap='coolwarm', s=5)
    # 调整布局并保存为单独的PDF文件
    plt.tight_layout()
    plt.savefig(f'{truncate_label(label)} vs Others.pdf', format='pdf')
    plt.close()  # 关闭图形以释放资源


#%% 重新尝试
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict


# 初始化独立热编码器
mlb = MultiLabelBinarizer(classes=NEW_LABELS)
mlb.fit([NEW_LABELS])  

features = []
labels = []

qualified_labels = set()

with torch.no_grad():
    for i, (imgs, target) in enumerate(train_loader):
        imgs = imgs.to(device)
        imgs = imgs[:, :14, :, :] if imgs.shape[1] > 14 else imgs

        feature = feature_model(imgs)
        batch_features = feature.view(feature.size(0), -1).cpu().numpy()
        batch_labels = [[NEW_LABELS[idx] for idx, label in enumerate(t) if label == 1] for t in target.cpu().numpy()]

        # 独立热编码标签
        batch_labels_encoded = mlb.fit_transform(batch_labels)

        features.append(batch_features)
        labels.append(batch_labels_encoded)  # 添加独立热编码后的标签数组

        print(f"After batch {i+1}: Features shape: {np.array(features).shape}, Labels collected: {len(labels)}")

features = np.vstack(features) if features else np.array([])
labels = np.vstack(labels) if labels else np.array([])  # 直接使用vstack堆叠

print("Accumulated features shape:", features.shape)
print("Number of labels collected:", labels.shape)


#%% PCA

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

# 将数据转换为 NumPy 数组（如果它们还不是）
if isinstance(features, torch.Tensor):
    features = features.cpu().numpy()  # 确保从 GPU 转移

# 将数据类型转换为 float32 减少内存消耗
import numpy as np
import gc

def process_and_save_batch(data, batch_idx):
    # 假设这是转换数据的函数
    data = data.astype(np.float32)  # 转换数据类型为 float32
    # 保存转换后的数据
    np.save(f'batch_{batch_idx}.npy', data)
    del data  # 删除数据以释放内存
    gc.collect()  # 垃圾回收

# 分成10个批次处理
n_batches = 10
batches = np.array_split(features, n_batches)

for i, batch in enumerate(batches):
    print(f"Processing batch {i+1}/{n_batches}")
    process_and_save_batch(batch, i)

# 重新加载和合并数据
accumulated_features = None
for i in range(n_batches):
    batch_data = np.load(f'batch_{i}.npy')
    if accumulated_features is None:
        accumulated_features = batch_data
    else:
        accumulated_features = np.concatenate([accumulated_features, batch_data])
    del batch_data  # 删除批次数据以释放内存
    print(f"Batch {i+1} loaded and concatenated.")

# 最后再次保存合并后的数据
np.save('processed_features.npy', features)


# 标准化数据
scaler = StandardScaler()
features = scaler.fit_transform(features)

print('Data normalization complete.')

# 使用 IncrementalPCA 来进行主成分分析
n_components = 5000  # 主成分数
n_batches = 100  # 分批处理的批次数量
ipca = IncrementalPCA(n_components=n_components)

# 分批进行PCA处理
for batch in np.array_split(features, n_batches):
    ipca.partial_fit(batch)

# 转换全部数据
features = ipca.transform(features)

print('PCA transformation complete.')

# 将降维后的数据转换回 PyTorch 张量，如果需要的话
features = torch.from_numpy(features).float()

print("Accumulated features shape:", features.shape)
print("Number of labels collected:", len(labels))

# 将features和labels保存为二进制格式的.npy文件
np.save('pca_features2.npy', features)  # 保存features数组
np.save('pca_labels2.npy', labels)      # 保存labels数组

print('Data saved successfully.')



#%% 保存数据结果
import numpy as np

# 将features和labels保存为二进制格式的.npy文件
np.save('features_try1.npy', features)  # 保存features数组
np.save('labels_try1.npy', labels)      # 保存labels数组

train_features = np.load('features_try1.npy',allow_pickle=True)
train_labels = np.load('labels_try1.npy',allow_pickle=True)
test_features = np.load('features_try2.npy',allow_pickle=True)
test_labels = np.load('labels_try2.npy',allow_pickle=True)
#%% 从.npy文件加载数据
import numpy as np
#train_features = np.load('principal_components.npy',allow_pickle=True)
train_features = np.load('train_features.npy',allow_pickle=True)
train_labels = np.load('train_labels.npy',allow_pickle=True)

test_features = np.load('test_features.npy',allow_pickle=True)
test_labels = np.load('test_labels.npy',allow_pickle=True)

#%% linear probing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


def hamming_loss(y_true, y_pred, threshold=0.5):
    # 将预测概率转换为0或1的预测结果
    y_pred = (y_pred > threshold).float()
    # 计算汉明损失：不一致的标签占总标签的比例
    loss = (y_true != y_pred).float().mean()
    return loss




train_labels_encoded = train_labels
test_labels_encoded = test_labels  # 使用相同的转换

# 转换为Tensor
train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels_encoded, dtype=torch.float32)
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels_encoded, dtype=torch.float32)

# 创建TensorDataset和DataLoader
train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=2000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2000, shuffle=False)

# 定义模型
class LinearModel(nn.Module):
    def __init__(self, num_features, num_labels):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(num_features, num_labels)

    def forward(self, x):
        return self.linear(x)  # BCEWithLogitsLoss includes the sigmoid activation

model = LinearModel(train_features_tensor.shape[1], train_labels_tensor.shape[1])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score

def evaluate_model(model, loader, device):
    macro_f2s, micro_f2s, hammings = [], [], []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        outputs_sig = torch.sigmoid(outputs).detach().cpu()
        targets = targets.cpu()

        macro_f2 = fbeta_score(targets.numpy(), (outputs_sig > 0.5).numpy(), beta=2, average='macro')
        micro_f2 = fbeta_score(targets.numpy(), (outputs_sig > 0.5).numpy(), beta=2, average='micro')
        hamming = hamming_loss(targets, outputs_sig )

        macro_f2s.append(macro_f2)
        micro_f2s.append(micro_f2)
        hammings.append(hamming.item())

    return {
        'macro_f2': np.mean(macro_f2s),
        'micro_f2': np.mean(micro_f2s),
        'hamming_loss': np.mean(hammings),
        'f2_sum': np.mean(macro_f2s) + np.mean(micro_f2s)
    }


train_f2_scores = []
test_f2_scores = []
train_hamming_losses = []
test_hamming_losses = []
best_f2_sum = float('-inf')
best_details = {}
test_macro_f2_scores = []
test_micro_f2_scores = []

model.train()
for epoch in range(100):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs_sig = torch.sigmoid(outputs)
        loss = nn.BCEWithLogitsLoss()(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_eval = evaluate_model(model, train_loader, device)
        test_eval = evaluate_model(model, test_loader, device)
        test_macro_f2_scores.append(test_eval['macro_f2'])
        test_micro_f2_scores.append(test_eval['micro_f2'])

        if test_eval['f2_sum'] > best_f2_sum:
            best_f2_sum = test_eval['f2_sum']
            best_details = {
                'epoch': epoch + 1,
                'test_macro_f2': test_eval['macro_f2'],
                'test_micro_f2': test_eval['micro_f2'],
                'test_hamming_loss': test_eval['hamming_loss']
            }

        train_f2_scores.append(train_eval['f2_sum'])
        test_f2_scores.append(test_eval['f2_sum'])
        train_hamming_losses.append(train_eval['hamming_loss'])
        test_hamming_losses.append(test_eval['hamming_loss'])

        print(f'Epoch {epoch+1}, Train/Test F2 sum: {train_eval["f2_sum"]:.4f}/{test_eval["f2_sum"]:.4f}, '
              f'Hamming Loss: {train_eval["hamming_loss"]:.4f}/{test_eval["hamming_loss"]:.4f}')

# 输出最优结果
print(f'Best F2 sum (Macro F2 + Micro F2): {best_f2_sum} at epoch {best_details["epoch"]}')
print(f'Macro F2: {best_details["test_macro_f2"]}, Micro F2: {best_details["test_micro_f2"]}, '
      f'Hamming Loss: {best_details["test_hamming_loss"]}')

# 可视化结果


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(test_macro_f2_scores, label='Test Macro F2 Score')
plt.plot(test_micro_f2_scores, label='Test Micro F2 Score')
plt.title('Test F2 Scores Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('F2 Score')
plt.legend()
plt.grid(True)


# 绘制汉明损失图表
plt.subplot(1, 2, 2)
plt.plot(test_hamming_losses, label='Test Hamming Loss', marker='o')
plt.title('Test Hamming Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Hamming Loss')
plt.legend()
plt.grid(True)
plt.savefig('linear_pca.pdf', format='pdf')

plt.tight_layout()
plt.show()

#%%随机森林forest
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, hamming_loss
import numpy as np
from sklearn.multioutput import MultiOutputClassifier


def hamming_loss(y_true, y_pred, threshold=0.5):
    # 假设 y_pred 是概率，需要先转换为二进制标签
    y_pred = (y_pred > threshold).astype(int)
    # 计算汉明损失
    return np.mean(y_true != y_pred)


# 自定义F2分数计分器
def f2_score_macro(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2, average='macro')

def f2_score_micro(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2, average='micro')


label_counts = np.sum(train_labels, axis=0)

# 计算权重
total_samples = train_labels.shape[0]
num_classes = train_labels.shape[1]

class_weights = [{i: total_samples / (num_classes * count) for i, count in enumerate(label_counts)} for _ in range(num_classes)]

# 随机森林模型（参数可根据需要调整）
rf_classifier = RandomForestClassifier(
    n_estimators=1,
    criterion='gini',
    max_depth=40,
    min_samples_split=10,
    min_samples_leaf=10,
    max_features=10000,
    max_leaf_nodes=10000,
    bootstrap=False,
    n_jobs=6,
    class_weight = class_weights
)

# 训练模型
rf_classifier.fit(train_features, train_labels)
y_pred = rf_classifier.predict(test_features)


# 计算F2分数和汉明损失
macro_f2 = f2_score_macro(test_labels, y_pred)
micro_f2 = f2_score_micro(test_labels, y_pred)
hamming_loss_value = hamming_loss(test_labels, y_pred)

print("Random Forest Hamming Loss:", hamming_loss_value)
print("Macro F2 Score：", macro_f2)
print("Micro F2 Score：", micro_f2)

#%% Resnet- 14channels

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models
from sklearn.metrics import fbeta_score
import numpy as np
from tqdm import tqdm

# 定义一个数据集类，用于处理和存储数据
class FixedDataset(Dataset):
    def __init__(self, loader, total_samples=15000, batch_size=3000, split_ratio=0.8):
        self.loader = loader
        self.features = []
        self.labels = []
        self.load_data(total_samples)
        self.train_batches, self.test_batches = self.create_batches(batch_size, split_ratio)

    def load_data(self, total_samples):
        count = 0
        progress_bar = tqdm(total=total_samples, desc="加载数据")
        for imgs, targets in self.loader:
            if count < total_samples:
                self.features.append(imgs[:, :14, :, :])  # 假设我们只需要前14个通道
                self.labels.append(targets)
                count += imgs.size(0)
                progress_bar.update(imgs.size(0))
            else:
                break
        progress_bar.close()
        self.features = torch.cat(self.features)[:total_samples]
        self.labels = torch.cat(self.labels)[:total_samples]
        print(f"总共加载了 {len(self.features)} 个样本")

    def create_batches(self, batch_size, split_ratio):
        # 划分数据集为训练集和测试集
        total_len = len(self.features)
        train_size = int(total_len * split_ratio)
        test_size = total_len - train_size

        # 创建训练数据批次
        train_indices = torch.arange(train_size).reshape(-1, batch_size)
        train_batches = [(self.features[idx], self.labels[idx]) for idx in train_indices]

        # 创建测试数据批次
        test_indices = torch.arange(train_size, total_len).reshape(-1, batch_size)
        test_batches = [(self.features[idx], self.labels[idx]) for idx in test_indices]

        return train_batches, test_batches

# 修改ResNet18以适应14通道输入
class ResNet18_14Channels(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(14, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# 初始化模型和参数

num_classes = 19  # 示例类别数
model = ResNet18_14Channels(num_classes=num_classes)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 实例化数据集
total_samples = 1500
batch_size = 100
dataset = FixedDataset(train_loader, total_samples, batch_size)

# 获取训练批次和测试批次
train_batches = dataset.train_batches
test_batches = dataset.test_batches

# 训练模型并记录损失和F2分数

f2_macro_scores = []
f2_micro_scores = []
hammingloss = []
f2_sum_scores = []  # 存储 F2 宏观和微观分数之和
best_f2_sum = 0  # 最优 F2 分数和的初始值
best_epoch = 0  # 用于记录取得最佳 F2 分数和的 epoch
best_test_loss = float('inf')  # 初始化最佳测试损失为无穷大
best_details = {}

def hamming_loss(y_true, y_pred, threshold=0.5):
    # 确保 y_pred 是 tensor
    y_pred = (y_pred > threshold).float()
    return (y_true != y_pred).float().mean()


import time
for epoch in range(500):  # 训练 10 个 epoch
    print(f"Starting epoch {epoch+1}")
    model.train()
    epoch_train_loss = 0
    for inputs, targets in train_batches:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    epoch_train_loss /= len(train_batches)
    print(f"Epoch {epoch+1}, Training Loss: {epoch_train_loss}")

    # 在测试集上评估模型
    # 在测试集上评估模型
    model.eval()
    epoch_test_loss = 0
    all_preds = []
    all_targets = []
    test_hamming_loss = 0

    with torch.no_grad():
        test_hamming_loss = 0
        for inputs, targets in test_batches:  # 使用测试批次
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_test_loss += loss.item()
            
            preds = torch.sigmoid(outputs)
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())
            start_time_gpu = time.time()
            current_hamming_loss = hamming_loss(targets, preds)
            test_hamming_loss += current_hamming_loss
            end_time_gpu = time.time()
            gpu_time = end_time_gpu - start_time_gpu
            print(f"GPU execution time: {gpu_time} seconds")
    epoch_test_loss /= len(test_batches)  # 计算平均测试损失
    all_preds_cpu = [pred.cpu().numpy() for pred in all_preds]
    all_preds = np.vstack(all_preds_cpu)
    all_targets = np.vstack(all_targets)
    macro_f2 = fbeta_score(all_targets, all_preds > 0.5, beta=2, average='macro')
    micro_f2 = fbeta_score(all_targets, all_preds > 0.5, beta=2, average='micro')
    f2_macro_scores.append(macro_f2)
    f2_micro_scores.append(micro_f2)
    f2_sum = macro_f2 + micro_f2
    f2_sum_scores.append(f2_sum)
    
    test_hamming_loss /= len(test_batches)
    print(f'Epoch {epoch+1}, Test Hamming Loss: {test_hamming_loss}')
    print(f'Epoch {epoch+1}, Training Loss: {epoch_train_loss}, Test Loss: {epoch_test_loss}, Macro F2: {macro_f2}, Micro F2: {micro_f2}, F2 Sum: {f2_sum}')
    hammingloss.append(test_hamming_loss)
    # 更新最优 F2 分数和
    if f2_sum > best_f2_sum:
        best_f2_sum = f2_sum
        best_epoch = epoch + 1
        best_details = {
            'epoch': epoch + 1,
            'macro_f2': macro_f2,
            'micro_f2': micro_f2,
            'hamming_loss': test_hamming_loss 
            }
        print(f"New best F2 sum: {best_f2_sum} at epoch {best_epoch}")
    
    # 早停判断
    if len(f2_sum_scores) > 10:
        recent_best_f2_sum = max(f2_sum_scores[-10:])
        previous_best_f2_sum = max(f2_sum_scores[:-10])
        if recent_best_f2_sum <= previous_best_f2_sum:
            print("Early stopping triggered due to lack of improvement in F2 sum over the last 10 epochs.")
            break

# 绘制损失和F2分数图
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.plot(f2_macro_scores, label='Macro F2 Score')
plt.plot(f2_micro_scores, label='Micro F2 Score')
plt.title('F2 Score Changes')
plt.xlabel('Epochs')
plt.ylabel('F2 Score')
plt.legend()

plt.subplot(1, 2, 2)
if isinstance(hammingloss[0], torch.Tensor):
    hammingloss = [hl.item() for hl in hammingloss]
plt.plot(hammingloss, label='Test Hamming Loss', marker='o')
plt.title('Training and Test Hamming Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Hamming Loss')
plt.legend()
plt.grid(True)


plt.show()

print(f'Best F2 sum (Macro F2 + Micro F2): {best_f2_sum} at epoch {best_epoch}')
print(f'Macro F2: {best_details["macro_f2"]}, Micro F2: {best_details["micro_f2"]}, Hamming Loss: {best_details["hamming_loss"]}')

'''
 torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()  # 清空CUDA缓存
'''