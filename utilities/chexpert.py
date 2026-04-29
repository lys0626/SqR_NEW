import os
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

class chexpert(Dataset):
    task = 'multilabel'
    num_labels = 13
    
    # 严格按照你提供的 CSV 表头顺序定义 14 个病理标签
    label_names = [
        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
    ]

    def __init__(self, root='/data/chexpert_224', mode='train', transform=None):
        self.root = root
        self.transform = transform
        self.mode = mode

        # 1. 定位 CSV 文件
        if mode == 'train':
            csv_file = 'train_8.csv' # 或者你的 train_8.csv
        elif mode == 'valid':
            csv_file = 'valid_2.csv'
        elif mode == 'test':
            csv_file = 'test_processed.csv' 
        else:
            raise ValueError(f"Unknown mode: {mode}")

        csv_path = os.path.join(self.root, csv_file)
        
        if not os.path.exists(csv_path):
             raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        print(f"Loading CheXpert dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)

        # 2. 直接提取 14 列预处理好的干净标签 (全部是 0.0 或 1.0)
        labels_np = self.df[self.label_names].values.astype(np.float32)

        # 3. 准备数据列表
        self.image_paths = []
        self.labels = []
        missing_count = 0
        
        for idx, row in self.df.iterrows():
            # 直接读取干净的 Path，例如 'train/patient00001/study1/view1_frontal.jpg'
            raw_path = str(row['Path'])
            
            final_path = os.path.join(self.root, raw_path)
            
            if os.path.exists(final_path):
                self.image_paths.append(final_path)
                self.labels.append(labels_np[idx])
            else:
                missing_count += 1
                if missing_count < 5: 
                     print(f"Warning: Image not found: {final_path}")

        if missing_count > 0:
            print(f"Total missing images in {mode} set: {missing_count}")
        else:
            print(f"Successfully loaded {len(self.image_paths)} images for {mode}.")

        self.y = np.array(self.labels)
        self.classes = self.label_names

        # 计算权重 (Stage 1 可选使用)
        if len(self.y) > 0:
            pos_counts = np.sum(self.y, axis=0)
            neg_counts = len(self.y) - pos_counts
            pos_counts = np.where(pos_counts == 0, 1, pos_counts)
            neg_counts = np.where(neg_counts == 0, 1, neg_counts)
            self.weight = np.stack([1.0 / neg_counts, 1.0 / pos_counts], axis=1)
        else:
            self.weight = np.ones((len(self.classes), 2))

    def get_number_classes(self):
        return self.num_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)
            
        # Stage 1 专用：返回图像, 标签, 和 idx (用于 FkL Mask)
        return image, label, idx