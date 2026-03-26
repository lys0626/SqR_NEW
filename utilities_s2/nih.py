import os
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import torch
class nihchest(Dataset):
    task = 'multilabel'
    num_labels = 14
    
    # 标签列名 (根据 cxr14_valid.csv 等文件确定)
    label_names = [
        'Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Mass', 'Pneumothorax',
        'Consolidation', 'Pleural_Thickening', 'Cardiomegaly', 'Emphysema', 'Edema',
        'Fibrosis', 'Pneumonia', 'Hernia'
    ]

    def __init__(self, root='/data/nih-chest-xrays', mode='train', transform=None,args=None):
        self.root = root
        self.transform = transform
        self.mode = mode

        # 1. 定位 CSV 文件
        # CSV 路径: /data/nih-chest-xrays/data_csv
        csv_root = os.path.join(self.root, 'data_csv')
        
        if mode == 'train':
            csv_file = 'cxr14_train.csv'
        elif mode == 'valid':
            csv_file = 'cxr14_valid.csv'
        elif mode == 'test':
            csv_file = 'cxr14_test.csv'
        else:
            raise ValueError(f"Unknown mode: {mode}")

        csv_path = os.path.join(csv_root, csv_file)
        
        if not os.path.exists(csv_path):
             raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        print(f"Loading dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)

        # 2. 准备数据列表
        self.image_paths = []
        self.labels = []
        self.global_indices = [] # 【新增】：用于追踪数据在 CSV 中的绝对行号，方便字典查表
        # 直接读取 14 列标签 (0/1格式)
        labels_np = self.df[self.label_names].values.astype(np.float32)
        
        # 指定图片文件夹路径
        img_folder = os.path.join(self.root, 'img_512')
        if not os.path.exists(img_folder):
            print(f"Warning: Image folder not found at {img_folder}!")

        missing_count = 0
        
        for idx, row in self.df.iterrows():
            # 获取文件名 (去除 CSV 中原有的绝对路径前缀)
            # 例如: /home/.../00005699_002.png -> 00005699_002.png
            raw_path = row['img_path']
            filename = os.path.basename(raw_path)
            
            # 构建新路径: /data/nih-chest-xrays/img_384/00005699_002.png
            final_path = os.path.join(img_folder, filename)
            
            # 检查文件是否存在 (可选，为了速度可以注释掉 if exists)
            if os.path.exists(final_path):
                self.image_paths.append(final_path)
                self.labels.append(labels_np[idx])
                self.global_indices.append(idx) # 记录真实的 idx
            else:
                # 记录缺失文件
                missing_count += 1
                if missing_count < 5: 
                     print(f"Warning: Image not found: {final_path}")

        if missing_count > 0:
            print(f"Total missing images in {mode} set: {missing_count}")
        else:
            print(f"Successfully loaded {len(self.image_paths)} images for {mode}.")

        self.y = np.array(self.labels)
        self.classes = self.label_names

        # 3. 计算正负样本权重 (Loss Balancing)
        if len(self.y) > 0:
            pos_counts = np.sum(self.y, axis=0)
            neg_counts = len(self.y) - pos_counts
            
            # 防止除零
            pos_counts = np.where(pos_counts == 0, 1, pos_counts)
            neg_counts = np.where(neg_counts == 0, 1, neg_counts)
            
            weight_pos = 1.0 / pos_counts
            weight_neg = 1.0 / neg_counts
            self.weight = np.stack([weight_neg, weight_pos], axis=1)
        else:
            self.weight = np.ones((len(self.classes), 2))
        # ================= 【新增】：加载 Stage 1 产出的交接文件 =================
        self.clean_indices = set()
        self.noisy_indices = set()
        self.noise_clean_labels_dict = {}
        self.cam_masks_dict = {}
        
        # 只有在 train 模式且 args 包含了 clean_mask_path 时才加载
        if self.mode == 'train' and args is not None and hasattr(args, 'clean_mask_path') and args.clean_mask_path:
            base_dir = os.path.dirname(args.clean_mask_path)
            
            clean_pt = os.path.join(base_dir, 'clean_indices.pt')
            if os.path.exists(clean_pt):
                self.clean_indices = set(torch.load(clean_pt))
                
            noisy_pt = os.path.join(base_dir, 'noisy_indices.pt')
            if os.path.exists(noisy_pt):
                self.noisy_indices = set(torch.load(noisy_pt))
                
            labels_pt = os.path.join(base_dir, 'noise_clean_labels_dict.pt')
            if os.path.exists(labels_pt):
                self.noise_clean_labels_dict = torch.load(labels_pt)
                
            masks_pt = os.path.join(base_dir, 'noise_cam_masks.pt')
            if os.path.exists(masks_pt):
                self.cam_masks_dict = torch.load(masks_pt)
        # ========================================================================
    def get_number_classes(self):
        return self.num_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        global_idx = self.global_indices[idx] # 拿出它真正的全局索引
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 简单容错：读取失败则尝试下一张
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)
            
        filename = os.path.basename(img_path)
        # ================= 【新增】：双轨身份识别与信号提纯 =================
        target = torch.tensor(label, dtype=torch.float32)
        
        if self.mode == 'train' and (self.clean_indices or self.noisy_indices):
            is_clean = (global_idx in self.clean_indices)
            
            if is_clean:
                # 干净样本，返回一个占位全 1 掩码
                cam_mask = torch.ones((2, 2), dtype=torch.bool)
            else:
                # 噪声样本：将 target 中被 Stage 1 鉴定为假阳性的标签强行置为 0
                if global_idx in self.noise_clean_labels_dict:
                    clean_label = self.noise_clean_labels_dict[global_idx]
                    if isinstance(clean_label, torch.Tensor):
                        clean_label = clean_label.cpu()
                    target = target * (clean_label > 0.5).float()
                
                # 获取它对应的物理拼接 2x2 掩码
                cam_mask = self.cam_masks_dict.get(global_idx, torch.zeros((2, 2), dtype=torch.bool))
        else:
            is_clean = True
            cam_mask = torch.ones((2, 2), dtype=torch.bool)
        # ========================================================================
        # 【严格保留原本的 data 字典格式，只向里面新增 key】
        data = {
            'image': image, 
            'target': target, 
            'name': filename,
            'is_clean': is_clean,
            'cam_mask': cam_mask
        }
        return data