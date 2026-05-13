import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from utilities.multilabel_noise import inject_multilabel_noise

class VinBigDataDataset(Dataset):
    task = 'multilabel'
    num_labels = 15
    
    label_names = [
        'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
        'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
        'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
        'Pulmonary fibrosis','No finding'
    ]

    def __init__(self, root='/data/dsj/lys/vinbigdata', mode='train', transform=None, 
                 inject_noise=False, noise_type='asym', sym_rate=0.2, fn_rate=0.4, fp_rate=0.1):
        self.root = root
        self.transform = transform
        self.mode = mode

        # 1. 定位 CSV 文件
        csv_root = self.root
        if mode == 'train':
            csv_file = 'train.csv'
        elif mode == 'valid':
            csv_file = 'valid.csv'
        elif mode == 'test':
            csv_file = 'vinbigdata_test.csv'
        else:
            raise ValueError(f"Unknown mode: {mode}")

        csv_path = os.path.join(csv_root, csv_file)
        if not os.path.exists(csv_path):
             raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        print(f"Loading VinBigData {mode} dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)

        # 2. 【关键步】首先提取干净的标签，用于后续注入噪声
        # 确保 labels_np 在被使用（注入噪声）之前已经定义
        labels_np = self.df[self.label_names].values.astype(np.float32)

        # 3. ================= 统一的噪声注入逻辑 =================
        if mode == 'train' and inject_noise:
            print(f"--> 为训练集注入 {noise_type} 多标签噪声...")
            
            # 调用我们在 multilabel_noise.py 中定义的工具
            # 注意：这里的噪声参数由外部传入，不使用未定义的 noise_rate
            noisy_labels_np, total_rate, n_type = inject_multilabel_noise(
                labels_np, 
                noise_type=noise_type, 
                sym_rate=sym_rate, 
                fn_rate=fn_rate, 
                fp_rate=fp_rate,
                no_finding_idx=None
            )
            
            # 动态生成文件名以区分不同的噪声设置
            if n_type == 'sym':
                noisy_filename = f'noisy_labels_SYM_{sym_rate}_Total{total_rate:.3f}.pt'
            else:
                noisy_filename = f'noisy_labels_ASYM_FN{fn_rate}_FP{fp_rate}_Total{total_rate:.3f}.pt'
            
            clean_filename = f'clean_labels_gt.pt'
            
            noisy_save_path = os.path.join(root, noisy_filename)
            clean_save_path = os.path.join(root, clean_filename)
            
            # 保存张量用于后期对比评测
            torch.save(torch.from_numpy(labels_np), clean_save_path)
            torch.save(torch.from_numpy(noisy_labels_np), noisy_save_path)
            
            print(f"--> [已保存] 干净真值: {clean_filename}")
            print(f"--> [已保存] 噪声输入: {noisy_filename}")
            
            # 将当前使用的标签更新为噪声标签
            self.labels = noisy_labels_np
        else:
            # 验证集或不注入噪声时，使用原始标签
            self.labels = labels_np

        # 4. 准备图片路径
        # 根据之前的讨论，你应该使用 resize 后的 img_224 文件夹提高训练速度
        img_folder = os.path.join(self.root, 'img_224') 
        if not os.path.exists(img_folder):
            # 如果 img_224 不存在，回退到原始 train 文件夹
            img_folder = os.path.join(self.root, 'train')

        self.image_paths = []
        final_labels = []
        missing_count = 0
        
        for idx, row in self.df.iterrows():
            filename = row['image_id'] + '.png'
            final_path = os.path.join(img_folder, filename)
            
            if os.path.exists(final_path):
                self.image_paths.append(final_path)
                # 将处理好的标签（可能是噪声，可能是干净）存入列表
                final_labels.append(self.labels[idx])
            else:
                missing_count += 1

        if missing_count > 0:
            print(f"Warning: Total missing images in {mode} set: {missing_count}")
        else:
            print(f"Successfully loaded {len(self.image_paths)} images for {mode}.")

        self.y = np.array(final_labels)
        self.classes = self.label_names

    def get_number_classes(self):
        return self.num_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.y[idx] # 使用 self.y 确保获取的是对应的（可能带噪声的）标签
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)
            
        return image, label, idx