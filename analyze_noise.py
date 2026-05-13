import argparse
import csv
import json
import os

import torch


DEFAULT_MASKS = (
    ("knn_clean_mask", "knn_clean_mask.pt"),
    ("knn_noisy_mask", "knn_noisy_mask.pt"),
    ("loss_clean_mask", "loss_clean_mask.pt"),
    ("loss_risk_mask", "loss_risk_mask.pt"),
    ("mee_easy_noisy_mask", "mee_easy_noisy_mask.pt"),
)


def load_tensor(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True).cpu()
    except TypeError:
        return torch.load(path, map_location="cpu").cpu()


def count(mask):
    return int(mask.sum().item())


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def load_label_names(path_or_csv, num_classes):
    if not path_or_csv:
        return [f"class_{idx}" for idx in range(num_classes)]
    if os.path.exists(path_or_csv):
        with open(path_or_csv, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        names = [item.strip() for item in path_or_csv.split(",") if item.strip()]
    if len(names) != num_classes:
        raise ValueError(f"Expected {num_classes} label names, got {len(names)} from {path_or_csv}")
    return names


def truth_masks(clean, noisy):
    return {
        "true_clean_positive": noisy & clean,
        "true_fp_noise": noisy & (~clean),
        "true_fn_noise": (~noisy) & clean,
        "true_clean_negative": (~noisy) & (~clean),
    }


def truth_type(clean_value, noisy_value):
    if noisy_value and clean_value:
        return "true_clean_positive"
    if noisy_value and not clean_value:
        return "true_fp_noise"
    if (not noisy_value) and clean_value:
        return "true_fn_noise"
    return "true_clean_negative"


def summarize_mask(mask_name, mask, truths):
    selected = count(mask)
    true_clean_positive = count(mask & truths["true_clean_positive"])
    true_fp_noise = count(mask & truths["true_fp_noise"])
    true_fn_noise = count(mask & truths["true_fn_noise"])
    true_clean_negative = count(mask & truths["true_clean_negative"])
    return {
        "mask": mask_name,
        "selected": selected,
        "true_clean_positive": true_clean_positive,
        "true_fp_noise": true_fp_noise,
        "true_fn_noise": true_fn_noise,
        "true_clean_negative": true_clean_negative,
        "positive_label_selected": true_clean_positive + true_fp_noise,
        "non_positive_label_selected": true_fn_noise + true_clean_negative,
        "clean_precision_among_selected": safe_div(true_clean_positive, selected),
        "fp_noise_rate_among_selected": safe_div(true_fp_noise, selected),
        "clean_recall_among_true_clean_positive": safe_div(true_clean_positive, count(truths["true_clean_positive"])),
        "fp_noise_recall_among_true_fp_noise": safe_div(true_fp_noise, count(truths["true_fp_noise"])),
    }


def per_class_rows(label_names, masks, truths):
    rows = []
    for mask_name, mask in masks.items():
        for c, label_name in enumerate(label_names):
            selected = mask[:, c]
            selected_count = count(selected)
            true_clean_positive = count(selected & truths["true_clean_positive"][:, c])
            true_fp_noise = count(selected & truths["true_fp_noise"][:, c])
            true_fn_noise = count(selected & truths["true_fn_noise"][:, c])
            true_clean_negative = count(selected & truths["true_clean_negative"][:, c])
            rows.append(
                {
                    "mask": mask_name,
                    "class": c,
                    "class_name": label_name,
                    "selected": selected_count,
                    "true_clean_positive": true_clean_positive,
                    "true_fp_noise": true_fp_noise,
                    "true_fn_noise": true_fn_noise,
                    "true_clean_negative": true_clean_negative,
                    "positive_label_selected": true_clean_positive + true_fp_noise,
                    "non_positive_label_selected": true_fn_noise + true_clean_negative,
                    "clean_precision_among_selected": safe_div(true_clean_positive, selected_count),
                    "fp_noise_rate_among_selected": safe_div(true_fp_noise, selected_count),
                    "class_true_clean_positive_total": count(truths["true_clean_positive"][:, c]),
                    "class_true_fp_noise_total": count(truths["true_fp_noise"][:, c]),
                    "clean_recall_in_class": safe_div(
                        true_clean_positive,
                        count(truths["true_clean_positive"][:, c]),
                    ),
                    "fp_noise_recall_in_class": safe_div(
                        true_fp_noise,
                        count(truths["true_fp_noise"][:, c]),
                    ),
                }
            )
    return rows


def entry_rows(label_names, clean, noisy, masks):
    rows = []
    for mask_name, mask in masks.items():
        indices = torch.nonzero(mask, as_tuple=False)
        for sample_idx, class_idx in indices.tolist():
            clean_value = bool(clean[sample_idx, class_idx].item())
            noisy_value = bool(noisy[sample_idx, class_idx].item())
            rows.append(
                {
                    "mask": mask_name,
                    "sample_index": sample_idx,
                    "class": class_idx,
                    "class_name": label_names[class_idx],
                    "noisy_label": int(noisy_value),
                    "clean_label": int(clean_value),
                    "truth_type": truth_type(clean_value, noisy_value),
                }
            )
    return rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_masks(args, expected_shape):
    mask_specs = list(DEFAULT_MASKS)
    for item in args.extra_mask:
        if "=" not in item:
            raise ValueError(f"--extra-mask must use name=path format, got: {item}")
        name, path = item.split("=", 1)
        mask_specs.append((name.strip(), path.strip()))

    masks = {}
    for name, filename_or_path in mask_specs:
        path = filename_or_path
        if args.stage1_dir and not os.path.isabs(path):
            path = os.path.join(args.stage1_dir, path)
        if not os.path.exists(path):
            if name in dict(DEFAULT_MASKS):
                print(f"Warning: missing {name}: {path}")
                continue
            raise FileNotFoundError(path)
        mask = load_tensor(path).bool()
        if mask.shape != expected_shape:
            raise ValueError(f"{name} shape {tuple(mask.shape)} != expected {tuple(expected_shape)}")
        masks[name] = mask
    if not masks:
        raise ValueError("No masks were loaded.")
    return masks


def print_summary(rows):
    print("=" * 88)
    print("Stage1 evidence-mask truth diagnosis")
    print("=" * 88)
    for row in rows:
        print(
            f"{row['mask']}: selected={row['selected']}, "
            f"true_clean={row['true_clean_positive']}, "
            f"true_fp_noise={row['true_fp_noise']}, "
            f"clean_precision={row['clean_precision_among_selected']:.2%}, "
            f"fp_noise_rate={row['fp_noise_rate_among_selected']:.2%}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Stage1 evidence masks against clean/noisy labels and list "
            "which selected labels are truly clean or truly noisy."
        )
    )
    parser.add_argument("--clean-labels", default="/data/dsj/lys/vinbigdata/clean_labels_gt.pt", help="Path to clean ground-truth labels.")
    parser.add_argument("--noisy-labels", default="/data/dsj/lys/vinbigdata/noisy_labels_ASYM_FN0.2_FP0.2_Total0.199.pt", help="Path to noisy labels used by Stage1.")
    parser.add_argument("--stage1-dir", default="/data/dsj/lys/SqR-NEW/experiment/VINVIG_denoise/5_12_0.94_0.95_4_asym_0.2_0.2_0.85_0.1_NEW_FP_MASK/vinbigdata/stage1_splicemix-cl", help="Directory containing Stage1 mask outputs.")
    parser.add_argument("--label-names", default="", help="Label txt path or comma-separated names.")
    parser.add_argument(
        "--extra-mask",
        action="append",
        default=[],
        help="Optional additional mask in name=path format. Relative paths are resolved under --stage1-dir.",
    )
    parser.add_argument("--summary-out", default="", help="Output summary JSON path.")
    parser.add_argument("--per-class-out", default="", help="Output per-class CSV path.")
    parser.add_argument("--entries-out", default="", help="Output selected label entries CSV path.")
    parser.add_argument(
        "--no-entries",
        action="store_true",
        default=False,
        help="Skip writing sample/class-level selected label entries.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    clean = load_tensor(args.clean_labels).bool()
    noisy = load_tensor(args.noisy_labels).bool()
    if noisy.shape != clean.shape:
        raise ValueError(f"noisy shape {tuple(noisy.shape)} != clean shape {tuple(clean.shape)}")

    label_names = load_label_names(args.label_names, clean.size(1))
    masks = load_masks(args, clean.shape)
    truths = truth_masks(clean, noisy)

    summary_rows = [summarize_mask(mask_name, mask, truths) for mask_name, mask in masks.items()]
    class_rows = per_class_rows(label_names, masks, truths)

    summary_out = args.summary_out or os.path.join(args.stage1_dir, "stage1_evidence_mask_summary.json")
    per_class_out = args.per_class_out or os.path.join(args.stage1_dir, "stage1_evidence_mask_per_class.csv")
    entries_out = args.entries_out or os.path.join(args.stage1_dir, "stage1_evidence_mask_entries.csv")

    print_summary(summary_rows)
    os.makedirs(os.path.dirname(os.path.abspath(summary_out)), exist_ok=True)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    write_csv(class_rows, per_class_out)
    print(f"Saved summary JSON: {summary_out}")
    print(f"Saved per-class CSV: {per_class_out}")

    if not args.no_entries:
        rows = entry_rows(label_names, clean, noisy, masks)
        write_csv(rows, entries_out)
        print(f"Saved selected label entries CSV: {entries_out}")


if __name__ == "__main__":
    main()