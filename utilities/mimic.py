import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd

class mimic(Dataset): 
    """
    适配 /data/mimic_cxr/PA/ 目录结构的 MIMIC 数据集类。
    读取 CSV 文件并加载 img_224 文件夹中的图片。
    """
    task = 'multilabel'
    
    # [重要] 这里必须使用生成 CSV 时所用的 13 个类别，顺序不能乱
    # 你的 CSV 文件是通过 cxr.py 的逻辑生成的，不包含 'No Finding'
    classes = [
        'Lung opacity', 'Pleural effusion', 'Atelectasis', 'Pneumonia', 
        'Cardiomegaly', 'Edema', 'Support devices', 'Lung lesion', 
        'Enlarged cardiomediastinum', 'Consolidation', 'Pneumothorax', 
        'Fracture', 'Pleural other'
    ]
    num_labels = len(classes) # 13

    def __init__(self, root='', mode='train', transform=None):
        """
        :param root: 数据集的根目录，例如 /data/mimic_cxr/PA
        :param mode: 'train', 'valid', 或 'test'
        """
        self.root = root
        self.transform = transform
        
        # 图片所在的文件夹名
        self.img_folder_name = 'img_224'
        self.img_root = os.path.join(self.root, self.img_folder_name)

        # 1. 确定要读取的 CSV 文件名
        if mode == 'train':
            csv_name = 'mimic_train_PA224.csv'
        elif mode == 'valid':
            # 你的文件在 /data/mimic_cxr/PA/ 下叫 mimic_val_PA224.csv
            csv_name = 'mimic_val_PA224.csv' 
        elif mode == 'test':
            csv_name = 'mimic_test_PA224.csv'
        else:
            raise ValueError(f"不支持的 mode: {mode}")
        
        csv_path = os.path.join(self.root, csv_name)
        
        # 2. 检查 CSV 是否存在
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件未找到: {csv_path}。请确认路径是否正确，文件是否在 /data/mimic_cxr/PA 下。")

        print(f"=> [{mode}] 正在加载 CSV: {csv_path}")
        
        # 3. 读取 CSV
        df = pd.read_csv(csv_path)
        
        # 4. 加载数据
        # CSV 中包含 relative img_path (如 p10/p.../xxx.jpg) 和标签列
        self.x = df['img_path'].tolist()
        self.y = df[self.classes].values.astype(np.float32)

        print(f"=> [{mode}] 成功加载 {len(self.x)} 条样本。")
        
        # 校验一下
        if self.y.shape[1] != self.num_labels:
             print(f"警告: CSV 中的列数 ({self.y.shape[1]}) 与代码定义的类别数 ({self.num_labels}) 不匹配！")
    
    def get_number_classes(self):
        return self.num_labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # CSV 里记录的路径，可能是 'p10/.../id.jpg'
        filename_record = self.x[idx] 
        label = self.y[idx]
        
        # --- [智能路径检测逻辑] ---
        # 你的 CSV 记录的是深层路径 (p10/p.../xxx.jpg)
        # 但你的 img_224 可能是扁平的 (只有图片)，也可能保留了结构。
        
        # 方式A: 尝试完整路径 (保留了目录结构) -> /data/mimic_cxr/PA/img_224/p10/.../xxx.jpg
        path_v1 = os.path.join(self.img_root, filename_record)
        
        # 方式B: 尝试仅文件名 (扁平目录结构) -> /data/mimic_cxr/PA/img_224/xxx.jpg
        path_v2 = os.path.join(self.img_root, os.path.basename(filename_record))

        if os.path.exists(path_v1):
            img_path = path_v1
        elif os.path.exists(path_v2):
            img_path = path_v2
        else:
            # 都找不到，打印错误并跳过
            # print(f"[Error] 图片未找到: {filename_record}")
            # print(f"尝试过: {path_v1} 和 {path_v2}")
            # 容错：返回下一个样本，避免程序崩溃
            return self.__getitem__((idx + 1) % len(self))

        try:
            # 强制转为 RGB (适配 ResNet)
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # 返回 engine.py 需要的字典格式
            # data = {'image': image, 'target': label, 'name': os.path.basename(filename_record)}
            return image, label, idx
        
        except Exception as e:
            print(f"加载图像出错 {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
