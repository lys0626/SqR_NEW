import json
import os
from glob import glob
from itertools import chain

import numpy as np
from PIL import Image
import pandas as pd
import torch # <--- 确保 torch 已导入
from torch.utils.data import Dataset

# 确保这个导入路径相对于您的项目结构是正确的
def load_json(file_path):
    with open(file_path, 'rt') as f:
        return json.load(f)

class chexpert(Dataset):
    """
    [已修改以兼容 SpliceMix 项目架构]
    """
    task = 'multilabel'
    
    # 注意: num_labels = 5 在 __init__ 中被 train_cols (长度为5) 隐式确认
    # 但 get_number_classes() 会返回 self.select_cols 的实际长度

    def __init__(self,
                 root='',
                 mode='train',
                 transform=None,
                 class_index=-1,
                 use_frontal=True,
                 use_upsampling=False,
                 flip_label=False,
                 verbose=False,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
                 **kwargs,
                 ):

        # --- [原始 __init__ 逻辑保持不变] ---
        
        # change test to valid b.c. no test split is reserved.
        # 确定要加载的 .csv 文件的名称
        filename_mode = mode
        if mode == 'test':
            mode = 'valid'  # 内部逻辑使用 'valid'
            filename_mode = 'test' # 但加载 'test.csv'
        elif mode == 'valid':
            # filename_mode = 'valid' # (原始代码) 为 'valid' 模式加载 'val.csv'
            # [!! 已修改 !!] 强制 'valid' 模式也加载 'test.csv'
            # 这符合 utils.py [cite: lys0626/resnet101/resnet101-fabe7a57d78a8cc899f4b7eb4d0e5558989465d2/utilities/utils.py] 中 'CHEXPERT' 的验证逻辑
            filename_mode = 'test'  

        # load data from csv
        self.classes = train_cols
        self.df = pd.read_csv(os.path.join(root, f'{filename_mode}.csv'))
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '', regex=False)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '', regex=False)
        if filename_mode == 'test':
            self.df['Path'] = self.df['Path'].str.replace('valid/', 'test/', regex=False)
        # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print('Upsampling %s...' % col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        # impute missing values
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ['No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity', 'Lung Lesion', 'Pneumonia',
                         'Pneumothorax', 'Pleural Other', 'Fracture', 'Support Devices']:  # other labels
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)

        self._num_images = len(self.df)

        if flip_label and class_index != -1:  # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)

        assert root != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1:  # 5 classes
            if verbose:
                print('Multi-label mode: True, Number of classes: [%d]' % len(train_cols))
                print('-' * 30)
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()

        self.class_index = class_index
        self.transform = transform

        self._images_list = [os.path.join(root, path) for path in self.df['Path'].tolist()]
        if class_index != -1:
            self.targets = self.df[train_cols].values[:, class_index].tolist()
        else:
            self.targets = self.df[train_cols].values.tolist()

        if verbose:
            if class_index != -1:
                if flip_label:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[-1] + self.value_counts_dict[1])
                    if verbose:
                        print('-' * 30)
                        print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1]))
                        print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                        print('-' * 30)
                else:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[0] + self.value_counts_dict[1])
                    if verbose:
                        print('-' * 30)
                        print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[0]))
                        print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                        print('-' * 30)
            else:
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    try:
                        imratio = self.value_counts_dict[class_key][1] / (
                                    self.value_counts_dict[class_key][0] + self.value_counts_dict[class_key][1])
                    except:
                        if len(self.value_counts_dict[class_key]) == 1:
                            only_key = list(self.value_counts_dict[class_key].keys())[0]
                            if only_key == 0:
                                self.value_counts_dict[class_key][1] = 0
                                imratio = 0  # no postive samples
                            else:
                                self.value_counts_dict[class_key][1] = 0
                                imratio = 1  # no negative samples

                    imratio_list.append(imratio)
                    if verbose:
                        print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0]))
                        print('%s(C%s): imbalance ratio is %.4f' % (select_col, class_key, imratio))
                        print()
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list

        pos_ratio = np.array(self.targets).mean(axis=0)
        self.weight = np.stack([pos_ratio, 1 - pos_ratio], axis=1)
        self.norm_weight = None

    # --- [修改开始] ---
    
    def get_number_classes(self):
        """
        返回数据集中的标签总数。
        utils.py [cite: lys0626/resnet101/resnet101-fabe7a57d78a8cc899f4b7eb4d0e5558989465d2/utilities/utils.py] 中的 get_dataset [cite: lys0626/resnet101/resnet101-fabe7a57d78a8cc899f4b7eb4d0e5558989465d2/utilities/utils.py] 会调用此方法。
        (此逻辑基于您原始文件中的 @property num_classes)
        """
        return len(self.select_cols)

    # (删除了 @property 装饰器的方法, 因为它们不再需要)

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        """
        获取单个数据样本。
        [已修改] 返回 engine.py [cite: lys0626/resnet101/resnet101-fabe7a57d78a8cc899f4b7eb4d0e5558989465d2/engine.py] 期望的字典格式。
        """
        img_path = self._images_list[idx]
        try:
            # 1. 强制转换为 'RGB' 以匹配 ResNet101 [cite: lys0626/resnet101/resnet101-fabe7a57d78a8cc899f4b7eb4d0e5558989465d2/models/ResNet_101.py] 的 3 通道输入
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            
            # 2. 获取标签 (逻辑与您的原始代码相同)
            if self.class_index != -1:  # multi-class mode
                label = np.array(self.targets[idx]).reshape(-1).astype(np.float32)
            else:
                label = np.array(self.targets[idx]).reshape(-1).astype(np.float32)
            
            # 3. 获取文件名
            filename = os.path.basename(img_path)

            # 4. 返回 engine.py [cite: lys0626/resnet101/resnet101-fabe7a57d78a8cc899f4b7eb4d0e5558989465d2/engine.py] 期望的字典
            data = {'image': image, 'target': label, 'name': filename}
            return data

        # 错误处理 (从 mimic.py [cite: lys0626/resnet101/resnet101-fabe7a57d78a8cc899f4b7eb4d0e5558989465d2/utilities/mimic.py] 借鉴)
        except FileNotFoundError:
            print(f"错误：图像文件未找到 {img_path}。正在跳过并加载下一个。")
            next_idx = (idx + 1) % len(self) if len(self) > 0 else 0
            if len(self) > 0: return self.__getitem__(next_idx)
            else: raise IndexError("数据集为空或无法加载任何图像")
        except Exception as e:
            print(f"加载图像 {img_path} 时出错：{e}。正在跳过并加载下一个。")
            next_idx = (idx + 1) % len(self) if len(self) > 0 else 0
            if len(self) > 0: return self.__getitem__(next_idx)
            else: raise IndexError("数据集为空或无法加载任何图像")
    
    # --- [修改结束] ---