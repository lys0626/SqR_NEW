import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utilities.multilabel_noise import inject_multilabel_noise
from utilities.padchest_xrv18 import PADCHEST_XRV16_LABELS


PADCHEST_METADATA_COLS = {
    "ImageID",
    "ImageDir",
    "PatientID",
    "StudyID",
    "ReportID",
    "StudyDate",
    "AcquisitionDate",
    "img_path",
    "MethodLabel",
    "Labels",
    "Localizations",
    "LabelsLocalizationsBySentence",
    "Report",
}


class padchest(Dataset):
    task = "multilabel"

    def __init__(
        self,
        root="",
        mode="train",
        transform=None,
        label_set="lt189",
        inject_noise=False,
        noise_type="asym",
        sym_rate=0.2,
        fn_rate=0.3,
        fp_rate=0.05,
        random_seed=42,
    ):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.label_set = label_set.lower()
        self.inject_noise = inject_noise
        self.noise_type = noise_type
        self.sym_rate = sym_rate
        self.fn_rate = fn_rate
        self.fp_rate = fp_rate
        self.random_seed = random_seed
        self._group_fallback_count = 0

        csv_path = self._resolve_csv_path(root, mode)
        print(f"Loading PadChest {self.label_set} {mode} dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.image_roots = self._load_image_roots(os.path.dirname(csv_path), root)

        self.classes = self._load_labels(os.path.dirname(csv_path), root)
        self.num_labels = len(self.classes)
        self.eval_classes = self._load_eval_labels(os.path.dirname(csv_path), root)
        self.eval_indices = [self.classes.index(label) for label in self.eval_classes if label in self.classes]

        if self.label_set == "xrv16":
            expected = len(PADCHEST_XRV16_LABELS)
            if self.num_labels != expected:
                print(f"Warning: expected {expected} XRV16 labels, got {self.num_labels}")
            labels_np = self.df[self.classes].values.astype(np.float32)
        else:
            if self.num_labels != 189:
                print(f"Warning: expected 189 PadChest-LT labels, got {self.num_labels}")
            labels_np = self.df[self.classes].values.astype(np.float32)

        self.image_paths = []
        self.labels = []
        self.group_keys = []
        self.row_indices = []
        missing_count = 0

        for idx, row in self.df.iterrows():
            img_path = self._resolve_image_path(row)
            if img_path is not None:
                self.image_paths.append(img_path)
                self.labels.append(labels_np[idx])
                self.group_keys.append(self._noise_group_key(row, idx))
                self.row_indices.append(idx)
            else:
                missing_count += 1
                if missing_count < 5:
                    print(f"Warning: PadChest image not found: {row.get('img_path', row.get('ImageID', idx))}")

        self.clean_labels = np.asarray(self.labels, dtype=np.float32)
        if self.clean_labels.size == 0:
            self.clean_labels = np.zeros((0, self.num_labels), dtype=np.float32)

        if mode == "train" and inject_noise:
            self.clean_labels, noisy_labels = self._inject_grouped_noise(self.clean_labels)
            self.labels = [row for row in noisy_labels]
            self.y = noisy_labels.astype(np.float32)
            self._save_noise_tensors(self.clean_labels, self.y)
        else:
            self.y = self.clean_labels.astype(np.float32)
            self.labels = [row for row in self.y]

        self._build_class_weights()

        if self._group_fallback_count > 0:
            print(
                "Warning: PadChest noise grouping fell back to ImageID for "
                f"{self._group_fallback_count} rows without ReportID/StudyID/PatientID+StudyDate."
            )
        if missing_count > 0:
            print(f"Total missing PadChest images in {mode} set: {missing_count}")
        print(f"Successfully loaded {len(self.image_paths)} PadChest images for {mode}.")

    def _resolve_csv_path(self, root, mode):
        csv_name = {"train": "train.csv", "valid": "valid.csv", "test": "test.csv"}.get(mode)
        if csv_name is None:
            raise ValueError(f"Unknown mode: {mode}")

        candidates = [
            os.path.join(root, csv_name),
            os.path.join(root, "data_csv", csv_name),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"PadChest CSV file not found. Tried: {candidates}")

    def _load_labels(self, csv_root, root):
        if self.label_set == "xrv16":
            for path in [
                os.path.join(csv_root, "padchest_xrv16_labels.txt"),
                os.path.join(root, "padchest_xrv16_labels.txt"),
            ]:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        labels = [line.strip() for line in f if line.strip()]
                    if labels:
                        return labels
            return list(PADCHEST_XRV16_LABELS)

        for path in [
            os.path.join(csv_root, "padchest_189_labels.txt"),
            os.path.join(root, "padchest_189_labels.txt"),
        ]:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return [line.strip() for line in f if line.strip()]

        return [col for col in self.df.columns if col not in PADCHEST_METADATA_COLS]

    def _load_eval_labels(self, csv_root, root):
        if self.label_set == "xrv16":
            return list(self.classes)

        for path in [
            os.path.join(csv_root, "padchest_170_eval_labels.txt"),
            os.path.join(root, "padchest_170_eval_labels.txt"),
        ]:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return [line.strip() for line in f if line.strip()]
        return list(self.classes)

    def _load_image_roots(self, csv_root, root):
        roots = []
        env_root = os.environ.get("PADCHEST_IMAGE_ROOT", "").strip()
        if env_root:
            roots.append(env_root)

        for path in [
            os.path.join(csv_root, "padchest_image_root.txt"),
            os.path.join(root, "padchest_image_root.txt"),
            os.path.join(csv_root, "image_root.txt"),
            os.path.join(root, "image_root.txt"),
        ]:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    image_root = f.readline().strip()
                if image_root:
                    roots.append(image_root)

        roots.extend([
            root,
            os.path.join(root, "images"),
            os.path.join(root, "img_224"),
            os.path.join(root, "images-224"),
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
        raw_path = str(row.get("img_path", "")).strip()
        image_id = str(row.get("ImageID", "")).strip()
        image_dir = str(row.get("ImageDir", "")).strip()

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
                if not os.path.splitext(image_id)[1]:
                    candidates.extend([
                        os.path.join(image_root, image_dir, f"{image_id}.png"),
                        os.path.join(image_root, f"{image_id}.png"),
                        os.path.join(image_root, image_dir, f"{image_id}.jpg"),
                        os.path.join(image_root, f"{image_id}.jpg"),
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

    def _noise_group_key(self, row, row_idx):
        for col in ("ReportID", "report_id", "ReportId"):
            value = self._clean_cell(row.get(col, ""))
            if value:
                return f"ReportID:{value}"
        for col in ("StudyID", "study_id", "StudyInstanceUID", "AccessionNumber"):
            value = self._clean_cell(row.get(col, ""))
            if value:
                return f"StudyID:{value}"

        patient_id = self._first_clean_cell(row, ("PatientID", "patient_id", "PatientID_DICOM"))
        study_date = self._first_clean_cell(row, ("StudyDate", "study_date", "AcquisitionDate", "StudyDate_DICOM"))
        if patient_id and study_date:
            return f"PatientStudy:{patient_id}:{study_date}"

        image_id = self._first_clean_cell(row, ("ImageID", "image_id", "img_path"))
        self._group_fallback_count += 1
        return f"ImageID:{image_id or row_idx}"

    def _first_clean_cell(self, row, names):
        for name in names:
            value = self._clean_cell(row.get(name, ""))
            if value:
                return value
        return ""

    @staticmethod
    def _clean_cell(value):
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except TypeError:
            pass
        value = str(value).strip()
        return "" if value.lower() in {"nan", "none", "null"} else value

    def _inject_grouped_noise(self, clean_labels):
        if clean_labels.size == 0:
            return clean_labels, clean_labels.copy()

        unique_keys = []
        key_to_idx = {}
        for key in self.group_keys:
            if key not in key_to_idx:
                key_to_idx[key] = len(unique_keys)
                unique_keys.append(key)

        group_clean = np.zeros((len(unique_keys), clean_labels.shape[1]), dtype=np.float32)
        for row_idx, key in enumerate(self.group_keys):
            group_clean[key_to_idx[key]] = np.maximum(group_clean[key_to_idx[key]], clean_labels[row_idx])

        no_finding_idx = None
        for candidate in ("No Finding", "No finding", "NoFinding"):
            if candidate in self.classes:
                no_finding_idx = self.classes.index(candidate)
                break

        group_noisy, total_rate, noise_name = inject_multilabel_noise(
            group_clean,
            noise_type=self.noise_type,
            sym_rate=self.sym_rate,
            fn_rate=self.fn_rate,
            fp_rate=self.fp_rate,
            random_seed=self.random_seed,
            no_finding_idx=no_finding_idx,
        )
        expanded_clean = np.asarray([group_clean[key_to_idx[key]] for key in self.group_keys], dtype=np.float32)
        expanded_noisy = np.asarray([group_noisy[key_to_idx[key]] for key in self.group_keys], dtype=np.float32)
        self._last_noise_rate = total_rate
        self._last_noise_name = noise_name
        print(f"--> Injected PadChest grouped noise on {len(unique_keys)} report/study groups.")
        return expanded_clean, expanded_noisy

    def _save_noise_tensors(self, clean_labels, noisy_labels):
        clean = torch.from_numpy(clean_labels.astype(np.float32))
        noisy = torch.from_numpy(noisy_labels.astype(np.float32))
        true_fp = (noisy == 1) & (clean == 0)
        true_fn = (noisy == 0) & (clean == 1)

        clean_path = os.path.join('/data/dsj/lys/padchest', "clean_labels_gt.pt")
        fp_path = os.path.join('/data/dsj/lys/padchest', "true_fp_noise_mask.pt")
        fn_path = os.path.join('/data/dsj/lys/padchest', "true_fn_noise_mask.pt")
        if getattr(self, "_last_noise_name", self.noise_type) == "sym":
            noisy_name = f"noisy_labels_SYM_{self.sym_rate}_Total{self._last_noise_rate:.3f}.pt"
        else:
            noisy_name = f"noisy_labels_ASYM_FN{self.fn_rate}_FP{self.fp_rate}_Total{self._last_noise_rate:.3f}.pt"
        noisy_path = os.path.join('/data/dsj/lys/padchest', noisy_name)

        torch.save(clean, clean_path)
        torch.save(noisy, noisy_path)
        torch.save(true_fp, fp_path)
        torch.save(true_fn, fn_path)
        print(f"--> [saved] clean labels: {clean_path}")
        print(f"--> [saved] noisy labels: {noisy_path}")
        print(f"--> [saved] true FP mask: {fp_path}")
        print(f"--> [saved] true FN mask: {fn_path}")

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
        label = self.y[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading PadChest image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        return image, label, idx
