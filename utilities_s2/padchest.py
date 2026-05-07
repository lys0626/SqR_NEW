import os
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class padchest(Dataset):
    task = 'multilabel'

    def __init__(self, root='', mode='train', transform=None):
        self.root = root
        self.transform = transform
        self.mode = mode

        csv_path = self._resolve_csv_path(root, mode)
        print(f"Loading PadChest-LT {mode} dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.image_roots = self._load_image_roots(os.path.dirname(csv_path), root)

        self.classes = self._load_labels(os.path.dirname(csv_path), root)
        
        # [修改点 1]: 补充全局的 num_labels 属性，适配 Stage 2 的 engine.py 和 loss 初始化
        self.num_labels = len(self.classes)  
        
        self.eval_classes = self._load_eval_labels(os.path.dirname(csv_path), root)
        self.eval_indices = [self.classes.index(label) for label in self.eval_classes if label in self.classes]

        if self.num_labels != 189:
            print(f"Warning: expected 189 PadChest-LT labels, got {self.num_labels}")

        labels_np = self.df[self.classes].values.astype(np.float32)
        self.image_paths = []
        self.labels = []
        missing_count = 0

        for idx, row in self.df.iterrows():
            img_path = self._resolve_image_path(row)
            if img_path is not None:
                self.image_paths.append(img_path)
                self.labels.append(labels_np[idx])
            else:
                missing_count += 1
                if missing_count < 5:
                    print(f"Warning: PadChest image not found: {row.get('img_path', row.get('ImageID', idx))}")

        self.y = np.asarray(self.labels, dtype=np.float32)
        if self.y.size == 0:
            self.y = np.zeros((0, self.num_labels), dtype=np.float32)

        self._build_class_weights()

        if missing_count > 0:
            print(f"Total missing PadChest images in {mode} set: {missing_count}")
        print(f"Successfully loaded {len(self.image_paths)} PadChest-LT images for {mode}.")

    def _resolve_csv_path(self, root, mode):
        csv_name = {'train': 'train.csv', 'valid': 'valid.csv', 'test': 'test.csv'}.get(mode)
        if csv_name is None:
            raise ValueError(f"Unknown mode: {mode}")

        candidates = [
            os.path.join(root, csv_name),
            os.path.join(root, 'data_csv', csv_name),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"PadChest-LT CSV file not found. Tried: {candidates}")

    def _load_labels(self, csv_root, root):
        for path in [
            os.path.join(csv_root, 'padchest_189_labels.txt'),
            os.path.join(root, 'padchest_189_labels.txt'),
        ]:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]

        metadata_cols = {'ImageID', 'ImageDir', 'PatientID', 'StudyID', 'ReportID', 'img_path', 'MethodLabel'}
        return [col for col in self.df.columns if col not in metadata_cols]

    def _load_eval_labels(self, csv_root, root):
        for path in [
            os.path.join(csv_root, 'padchest_170_eval_labels.txt'),
            os.path.join(root, 'padchest_170_eval_labels.txt'),
        ]:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]
        return list(self.classes)

    def _load_image_roots(self, csv_root, root):
        roots = []
        env_root = os.environ.get('PADCHEST_IMAGE_ROOT', '').strip()
        if env_root:
            roots.append(env_root)

        for path in [
            os.path.join(csv_root, 'padchest_image_root.txt'),
            os.path.join(root, 'padchest_image_root.txt'),
            os.path.join(csv_root, 'image_root.txt'),
            os.path.join(root, 'image_root.txt'),
        ]:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    image_root = f.readline().strip()
                if image_root:
                    roots.append(image_root)

        # [修改点 2]: 增加 'images-224' 目录查找
        roots.extend([
            root, 
            os.path.join(root, 'images'), 
            os.path.join(root, 'img_224'),
            os.path.join(root, 'images-224')
        ])

        unique_roots = []
        seen = set()
        for image_root in roots:
            image_root = os.path.normpath(image_root)
            if image_root not in seen:
                seen.add(image_root)
                unique_roots.append(image_root)
        return unique_roots

    def _candidate_image_paths(self, row):
        raw_path = str(row.get('img_path', '')).strip()
        image_id = str(row.get('ImageID', '')).strip()
        image_dir = str(row.get('ImageDir', '')).strip()

        candidates = []
        if raw_path:
            if os.path.isabs(raw_path):
                candidates.append(raw_path)
            for image_root in self.image_roots:
                candidates.append(os.path.join(image_root, raw_path))

        if image_id:
            for image_root in self.image_roots:
                candidates.extend([
                    os.path.join(image_root, image_dir, image_id),
                    os.path.join(image_root, image_id),
                ])

        seen = set()
        for path in candidates:
            if path and path not in seen:
                seen.add(path)
                yield path

    def _resolve_image_path(self, row):
        for path in self._candidate_image_paths(row):
            if os.path.exists(path):
                return path
        return None

    def _build_class_weights(self):
        if len(self.y) > 0:
            pos_counts = np.sum(self.y, axis=0)
            neg_counts = len(self.y) - pos_counts
            pos_counts = np.where(pos_counts == 0, 1, pos_counts)
            neg_counts = np.where(neg_counts == 0, 1, neg_counts)
            self.weight = np.stack([1.0 / neg_counts, 1.0 / pos_counts], axis=1)
        else:
            self.weight = np.ones((self.num_labels, 2))

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
            print(f"Error loading PadChest image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        filename = os.path.basename(img_path)
        
        # [关键]: Stage 2 专用的字典返回格式，包含 filename 键值
        return {'image': image, 'target': label, 'name': filename}