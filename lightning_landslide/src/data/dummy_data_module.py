"""
虚拟数据模块 - 简化版

为了快速测试，这是一个最小化的虚拟数据模块实现。
"""

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple

class DummyLandslideDataset(Dataset):
    def __init__(self, num_samples: int = 100, input_channels: int = 5, 
                 image_size: int = 64, positive_ratio: float = 0.1):
        self.num_samples = num_samples
        self.input_channels = input_channels
        self.image_size = image_size
        
        # 生成简单的随机数据
        self.data = torch.randn(num_samples, input_channels, image_size, image_size)
        
        # 生成标签
        num_positive = int(num_samples * positive_ratio)
        labels = torch.cat([
            torch.ones(num_positive),
            torch.zeros(num_samples - num_positive)
        ])
        
        # 打乱顺序
        perm = torch.randperm(num_samples)
        self.labels = labels[perm].float()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DummyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_samples: int = 100,
                 input_channels: int = 5, image_size: int = 64, 
                 num_workers: int = 0, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.input_channels = input_channels
        self.image_size = image_size
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            full_dataset = DummyLandslideDataset(
                num_samples=self.num_samples,
                input_channels=self.input_channels,
                image_size=self.image_size
            )
            
            train_size = int(0.8 * self.num_samples)
            val_size = self.num_samples - train_size
            
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = DummyLandslideDataset(
                num_samples=min(50, self.num_samples // 2),
                input_channels=self.input_channels,
                image_size=self.image_size
            )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                         shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                         shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                         shuffle=False, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return self.test_dataloader()
