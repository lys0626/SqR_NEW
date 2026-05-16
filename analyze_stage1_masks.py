import argparse
import csv
import os

import torch


def load_tensor(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True).cpu()
    except TypeError:
        return torch.load(path, map_location="cpu").cpu()


def pct(num, den):
    if den == 0:
        return "n/a"
    return f"{num / den:.2%}"


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def count(mask):
    return int(mask.sum().item())


def metric_line(name, selected, truth, clean_counterpart):
    selected_count = count(selected)
    truth_count = count(truth)
    tp = count(selected & truth)
    fp = count(selected & clean_counterpart)
    missed = count((~selected) & truth)
    print(f"{name}:")
    print(f"  Stage1 选中的标签数: {selected_count}")
    print(f"  该分支真实噪声标签数: {truth_count}")
    print(f"  选中且确实为噪声的标签数: {tp}")
    print(f"  误选到原本干净标签的数量: {fp}")
    print(f"  真实噪声但没有被选中的数量: {missed}")
    print(f"  选择准确率 precision: {pct(tp, selected_count)}")
    print(f"  噪声召回率 recall: {pct(tp, truth_count)}")
    print()


def per_class_rows(clean, noisy, soft_pred, fp_mask, fn_mask):
    clean = clean.bool()
    noisy = noisy.bool()
    soft_pred = soft_pred.bool()
    fp_mask = fp_mask.bool()
    fn_mask = fn_mask.bool()

    true_fp_noise = noisy & (~clean)
    true_fn_noise = (~noisy) & clean
    clean_pos = noisy & clean
    clean_neg = (~noisy) & (~clean)
    degraded_clean_pos = clean_pos & (~soft_pred)
    degraded_clean_neg = clean_neg & soft_pred

    rows = []
    for c in range(clean.size(1)):
        fp_selected = fp_mask[:, c]
        fn_selected = fn_mask[:, c]
        fp_truth = true_fp_noise[:, c]
        fn_truth = true_fn_noise[:, c]
        row = {
            "class": c,
            "noisy_pos": count(noisy[:, c]),
            "noisy_neg": count(~noisy[:, c]),
            "true_fp_noise": count(fp_truth),
            "true_fn_noise": count(fn_truth),
            "fp_selected": count(fp_selected),
            "fp_tp": count(fp_selected & fp_truth),
            "fp_false_clean_pos": count(fp_selected & clean_pos[:, c]),
            "fp_precision": safe_div(count(fp_selected & fp_truth), count(fp_selected)),
            "fp_recall": safe_div(count(fp_selected & fp_truth), count(fp_truth)),
            "fn_selected": count(fn_selected),
            "fn_tp": count(fn_selected & fn_truth),
            "fn_false_clean_neg": count(fn_selected & clean_neg[:, c]),
            "fn_precision": safe_div(count(fn_selected & fn_truth), count(fn_selected)),
            "fn_recall": safe_div(count(fn_selected & fn_truth), count(fn_truth)),
            "clean_pos_degraded": count(degraded_clean_pos[:, c]),
            "clean_pos_degraded_in_fp_mask": count(degraded_clean_pos[:, c] & fp_selected),
            "clean_neg_degraded": count(degraded_clean_neg[:, c]),
            "clean_neg_degraded_in_fn_mask": count(degraded_clean_neg[:, c] & fn_selected),
        }
        rows.append(row)
    return rows


def print_top(rows, key, title, n=8):
    ranked = sorted(rows, key=lambda row: row[key], reverse=True)
    print(title)
    for row in ranked[:n]:
        print(
            f"  class {row['class']:>2}: {key}={row[key]}, "
            f"fp_sel={row['fp_selected']}, fn_sel={row['fn_selected']}, "
            f"fp_prec={row['fp_precision']:.3f}, fn_prec={row['fn_precision']:.3f}"
        )
    print()


def print_four_way_summary(clean, noisy, soft_pred, fp_mask, fn_mask):
    true_fp_noise = noisy & (~clean)
    true_fn_noise = (~noisy) & clean
    clean_pos = noisy & clean
    clean_neg = (~noisy) & (~clean)

    fp_corrected = true_fp_noise & (~soft_pred)
    fp_not_corrected = true_fp_noise & soft_pred
    fn_corrected = true_fn_noise & soft_pred
    fn_not_corrected = true_fn_noise & (~soft_pred)

    fp_branch_wrong_selected = fp_mask & clean_pos
    fn_branch_wrong_selected = fn_mask & clean_neg
    fp_direction_wrong_final = clean_pos & (~soft_pred)
    fn_direction_wrong_final = clean_neg & soft_pred

    print("四象限纠错结果汇总")
    print("  假阳性标签纠错结果：原本为阴性，但注入噪声后被标成阳性")
    print(f"    注入噪声后假阳性标签总数: {count(true_fp_noise)}")
    print(f"    Stage1 成功纠正回阴性的数量: {count(fp_corrected)} / {count(true_fp_noise)} ({pct(count(fp_corrected), count(true_fp_noise))})")
    print(f"    Stage1 没有纠正回来、仍被视为阳性的数量: {count(fp_not_corrected)} / {count(true_fp_noise)} ({pct(count(fp_not_corrected), count(true_fp_noise))})")
    print("  假阴性标签纠错结果：原本为阳性，但注入噪声后被标成阴性")
    print(f"    注入噪声后假阴性标签总数: {count(true_fn_noise)}")
    print(f"    Stage1 成功纠正回阳性的数量: {count(fn_corrected)} / {count(true_fn_noise)} ({pct(count(fn_corrected), count(true_fn_noise))})")
    print(f"    Stage1 没有纠正回来、仍被视为阴性的数量: {count(fn_not_corrected)} / {count(true_fn_noise)} ({pct(count(fn_not_corrected), count(true_fn_noise))})")
    print("  原本干净标签被错误修改的结果")
    print(f"    原本干净阳性标签被 fp_mask 选中降权的数量: {count(fp_branch_wrong_selected)}")
    print(f"    原本干净阴性标签被 fn_mask 选中挖掘的数量: {count(fn_branch_wrong_selected)}")
    print(f"    最终被错误改成阴性的干净阳性标签数: {count(fp_direction_wrong_final)}")
    print(f"    最终被错误改成阳性的干净阴性标签数: {count(fn_direction_wrong_final)}")
    print()


def print_no_finding_summary(clean, noisy, soft_pred, fp_mask, fn_mask, no_finding_idx):
    if no_finding_idx < 0 or no_finding_idx >= clean.size(1):
        raise ValueError(
            f"no_finding_idx={no_finding_idx} is out of range for "
            f"{clean.size(1)} classes."
        )

    clean_nf = clean[:, no_finding_idx].bool()
    noisy_nf = noisy[:, no_finding_idx].bool()
    soft_pred_nf = soft_pred[:, no_finding_idx].bool()
    fp_mask_nf = fp_mask[:, no_finding_idx].bool()
    fn_mask_nf = fn_mask[:, no_finding_idx].bool()

    nf_changed = clean_nf != noisy_nf
    nf_fp_noise = noisy_nf & (~clean_nf)
    nf_fn_noise = (~noisy_nf) & clean_nf

    nf_fp_mask_tp = fp_mask_nf & nf_fp_noise
    nf_fn_mask_tp = fn_mask_nf & nf_fn_noise
    nf_mask_tp = nf_fp_mask_tp | nf_fn_mask_tp

    nf_fp_corrected = nf_fp_noise & (~soft_pred_nf)
    nf_fn_corrected = nf_fn_noise & soft_pred_nf
    nf_final_recovered = nf_changed & (soft_pred_nf == clean_nf)
    nf_final_not_recovered = nf_changed & (soft_pred_nf != clean_nf)

    nf_fp_wrong_selected_clean = fp_mask_nf & noisy_nf & clean_nf
    nf_fn_wrong_selected_clean = fn_mask_nf & (~noisy_nf) & (~clean_nf)

    print("No finding 类专项分析")
    print(f"  No finding 类索引: {no_finding_idx}")
    print(f"  干净标签中 No finding 阳性数: {count(clean_nf)}")
    print(f"  噪声标签中 No finding 阳性数: {count(noisy_nf)}")
    print(f"  No finding 被噪声改变总数: {count(nf_changed)}")
    print(f"    被错误加入 0->1 的数量: {count(nf_fp_noise)}")
    print(f"    被错误删除 1->0 的数量: {count(nf_fn_noise)}")
    print("  Stage1 mask 对 No finding 改变的找回情况")
    print(f"    fp_mask 命中 0->1 错误: {count(nf_fp_mask_tp)} / {count(nf_fp_noise)} ({pct(count(nf_fp_mask_tp), count(nf_fp_noise))})")
    print(f"    fn_mask 命中 1->0 错误: {count(nf_fn_mask_tp)} / {count(nf_fn_noise)} ({pct(count(nf_fn_mask_tp), count(nf_fn_noise))})")
    print(f"    mask 合计命中被改变标签: {count(nf_mask_tp)} / {count(nf_changed)} ({pct(count(nf_mask_tp), count(nf_changed))})")
    print(f"    fp_mask 误选 No finding 干净阳性: {count(nf_fp_wrong_selected_clean)}")
    print(f"    fn_mask 误选 No finding 干净阴性: {count(nf_fn_wrong_selected_clean)}")
    print("  soft label 二值化后 No finding 的最终找回情况")
    print(f"    0->1 错误最终被纠正回 0: {count(nf_fp_corrected)} / {count(nf_fp_noise)} ({pct(count(nf_fp_corrected), count(nf_fp_noise))})")
    print(f"    1->0 错误最终被纠正回 1: {count(nf_fn_corrected)} / {count(nf_fn_noise)} ({pct(count(nf_fn_corrected), count(nf_fn_noise))})")
    print(f"    所有被改变的 No finding 中最终恢复到干净标签: {count(nf_final_recovered)} / {count(nf_changed)} ({pct(count(nf_final_recovered), count(nf_changed))})")
    print(f"    所有被改变的 No finding 中最终仍未恢复: {count(nf_final_not_recovered)} / {count(nf_changed)} ({pct(count(nf_final_not_recovered), count(nf_changed))})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Stage1 fp_mask.pt and fn_mask.pt against clean/noisy labels."
    )
    parser.add_argument(
        "--clean-labels",
        default="/data/dsj/lys/vinbigdata/clean_labels_gt.pt",
        help="Path to clean ground-truth labels.",
    )
    parser.add_argument(
        "--noisy-labels",
        default="/data/dsj/lys/vinbigdata/noisy_labels_ASYM_FN0.2_FP0.2_Total0.200.pt",
        help="Path to injected-noise labels used by Stage1.",
    )
    parser.add_argument("--soft-targets", default="/data/dsj/lys/SqR-NEW/experiment/VINVIG_denoise/NEW_FN/best/5_15_asym_0.05_0.5_0.05_0.6_0.85_0.1/best/stage1_splicemix-cl/asymmetric_soft_targets.pt")
    parser.add_argument("--fp-mask", default="/data/dsj/lys/SqR-NEW/experiment/VINVIG_denoise/NEW_FN/best/5_15_asym_0.05_0.5_0.05_0.6_0.85_0.1/best/stage1_splicemix-cl/fp_mask.pt")
    parser.add_argument("--fn-mask", default="/data/dsj/lys/SqR-NEW/experiment/VINVIG_denoise/NEW_FN/best/5_15_asym_0.05_0.5_0.05_0.6_0.85_0.1/best/stage1_splicemix-cl/fn_mask.pt")
    parser.add_argument("--threshold", default=0.7, type=float)
    parser.add_argument("--no-finding-idx", default=14, type=int)
    args = parser.parse_args()

    soft_path = args.soft_targets
    fp_path = args.fp_mask
    fn_path = args.fn_mask
    clean = load_tensor(args.clean_labels).bool()
    noisy = load_tensor(args.noisy_labels).bool()
    soft = load_tensor(soft_path).float()
    fp_mask = load_tensor(fp_path).bool()
    fn_mask = load_tensor(fn_path).bool()

    expected_shape = clean.shape
    for name, tensor in {
        "noisy": noisy,
        "soft": soft,
        "fp_mask": fp_mask,
        "fn_mask": fn_mask,
    }.items():
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != clean shape {tuple(expected_shape)}")

    soft_pred = soft > args.threshold
    true_fp_noise = noisy & (~clean)
    true_fn_noise = (~noisy) & clean
    clean_pos = noisy & clean
    clean_neg = (~noisy) & (~clean)

    print("=" * 72)
    print("Stage1 标签纠错诊断")
    print("=" * 72)
    print(f"标签总数: {clean.numel()}")
    print(f"注入噪声后假阳性标签数: {count(true_fp_noise)}")
    print(f"注入噪声后假阴性标签数: {count(true_fn_noise)}")
    print(f"注入噪声总数: {count(noisy != clean)}")
    print()

    print_no_finding_summary(clean, noisy, soft_pred, fp_mask, fn_mask, args.no_finding_idx)

    metric_line("fp_mask 对假阳性标签的选择质量", fp_mask, true_fp_noise, clean_pos)
    metric_line("fn_mask 对假阴性标签的选择质量", fn_mask, true_fn_noise, clean_neg)
    print_four_way_summary(clean, noisy, soft_pred, fp_mask, fn_mask)

    corrected_fp = true_fp_noise & (~soft_pred)
    corrected_fn = true_fn_noise & soft_pred
    degraded_clean_pos = clean_pos & (~soft_pred)
    degraded_clean_neg = clean_neg & soft_pred
    final_errors = soft_pred != clean

    print("软标签二值化后的最终效果")
    print(f"  二值化阈值: {args.threshold}")
    print(f"  最终硬标签错误数: {count(final_errors)} ({pct(count(final_errors), clean.numel())})")
    print(f"  注入噪声后的假阳性标签中，被成功纠正回阴性的数量: {count(corrected_fp)} / {count(true_fp_noise)} ({pct(count(corrected_fp), count(true_fp_noise))})")
    print(f"  注入噪声后的假阴性标签中，被成功纠正回阳性的数量: {count(corrected_fn)} / {count(true_fn_noise)} ({pct(count(corrected_fn), count(true_fn_noise))})")
    print(f"  原本干净阳性标签被错误改成阴性的数量: {count(degraded_clean_pos)}")
    print(f"    其中落在 fp_mask 中的数量: {count(degraded_clean_pos & fp_mask)}")
    print(f"  原本干净阴性标签被错误改成阳性的数量: {count(degraded_clean_neg)}")
    print(f"    其中落在 fn_mask 中的数量: {count(degraded_clean_neg & fn_mask)}")
    print()


if __name__ == "__main__":
    main()
