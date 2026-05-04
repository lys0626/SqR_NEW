import csv
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture


def _empty_bool_like(targets):
    return torch.zeros_like(targets, dtype=torch.bool)


def _fit_clean_component(values, min_count, seed, prob_thresh):
    if len(values) < min_count:
        return np.ones(len(values), dtype=bool)

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = values.reshape(-1, 1)
    try:
        gmm = GaussianMixture(
            n_components=2,
            max_iter=50,
            tol=1e-3,
            reg_covar=1e-3,
            random_state=seed,
        )
        gmm.fit(values)
        clean_comp = gmm.means_.argmin()
        clean_prob = gmm.predict_proba(values)[:, clean_comp]
        return clean_prob > prob_thresh
    except Exception:
        return np.ones(len(values), dtype=bool)


def loss_gmm_split(losses, targets, min_pos=100, prob_thresh=0.5, seed=95):
    """Per-class log-loss GMM split for positive labels."""
    targets = targets.bool()
    loss_clean = _empty_bool_like(targets)
    loss_risk = _empty_bool_like(targets)

    for c in range(targets.size(1)):
        pos = torch.where(targets[:, c])[0]
        if len(pos) == 0:
            continue

        class_losses = losses[pos, c].detach().float().cpu().numpy()
        log_losses = np.log(class_losses + 1e-8)
        if len(log_losses) >= min_pos:
            p1, p99 = np.percentile(log_losses, 1), np.percentile(log_losses, 99)
            log_losses = np.clip(log_losses, p1, p99)

        is_clean = _fit_clean_component(log_losses, min_pos, seed, prob_thresh)
        is_clean = torch.from_numpy(is_clean).bool()
        clean_idx = pos[is_clean.to(pos.device)]
        risk_idx = pos[(~is_clean).to(pos.device)]
        loss_clean[clean_idx, c] = True
        loss_risk[risk_idx, c] = True

    return loss_clean, loss_risk


def top_loss_mask(losses, candidate_mask, top_ratio=0.3):
    selected = _empty_bool_like(candidate_mask)
    if top_ratio <= 0:
        return selected

    for c in range(candidate_mask.size(1)):
        idx = torch.where(candidate_mask[:, c])[0]
        if len(idx) == 0:
            continue
        k = max(1, int(math.ceil(len(idx) * top_ratio)))
        values = losses[idx, c]
        top_local = torch.topk(values, k=min(k, len(idx)), largest=True).indices
        selected[idx[top_local], c] = True

    return selected


def prototype_gmm_evidence(class_features, targets, seed_clean_mask, min_pos=100, prob_thresh=0.5, seed=95):
    """Build per-class prototypes from seed-clean positives and GMM split positive-label distances."""
    device = class_features.device
    targets = targets.bool().to(device)
    seed_clean_mask = seed_clean_mask.bool().to(device)
    num_samples, num_classes, _ = class_features.shape

    proto_clean = torch.zeros((num_samples, num_classes), dtype=torch.bool, device=device)
    proto_noisy = torch.zeros_like(proto_clean)
    proto_sim = torch.zeros((num_samples, num_classes), dtype=torch.float32, device=device)

    for c in range(num_classes):
        pos = torch.where(targets[:, c])[0]
        if len(pos) == 0:
            continue

        seed_idx = torch.where(seed_clean_mask[:, c] & targets[:, c])[0]
        if len(seed_idx) < 2:
            seed_idx = pos

        feats_c = F.normalize(class_features[:, c, :].float(), p=2, dim=1)
        proto = F.normalize(feats_c[seed_idx].mean(dim=0), p=2, dim=0)
        sims = torch.mv(feats_c, proto)
        proto_sim[:, c] = sims

        dists = (1.0 - sims[pos]).detach().cpu().numpy()
        is_clean = _fit_clean_component(dists, min_pos, seed, prob_thresh)
        is_clean = torch.from_numpy(is_clean).bool().to(device)
        proto_clean[pos[is_clean], c] = True
        proto_noisy[pos[~is_clean], c] = True

    return proto_clean, proto_noisy, proto_sim


def _knn_threshold(pos_count, tail_cutoff, mid_cutoff, tail_purity, mid_purity, head_purity):
    if pos_count < tail_cutoff:
        return tail_purity
    if pos_count < mid_cutoff:
        return mid_purity
    return head_purity


def knn_purity_evidence(
    class_features,
    targets,
    support_mask,
    k=50,
    chunk_size=512,
    tail_cutoff=100,
    mid_cutoff=200,
    tail_purity=0.0,
    mid_purity=0.05,
    head_purity=0.1,
):
    """Compute local label-support purity in each class-specific feature space."""
    device = class_features.device
    targets = targets.bool().to(device)
    support_mask = support_mask.bool().to(device)
    num_samples, num_classes, _ = class_features.shape

    knn_clean = torch.zeros((num_samples, num_classes), dtype=torch.bool, device=device)
    knn_noisy = torch.zeros_like(knn_clean)
    purity_scores = torch.zeros((num_samples, num_classes), dtype=torch.float32, device=device)

    if k <= 0:
        knn_clean[targets] = True
        return knn_clean, knn_noisy, purity_scores

    for c in range(num_classes):
        pos = torch.where(targets[:, c])[0]
        pos_count = len(pos)
        if pos_count == 0:
            continue

        threshold = _knn_threshold(
            pos_count,
            tail_cutoff,
            mid_cutoff,
            tail_purity,
            mid_purity,
            head_purity,
        )
        feats_c = F.normalize(class_features[:, c, :].float(), p=2, dim=1)
        support = support_mask[:, c]
        if int(support.sum().item()) < max(1, min(k, pos_count) // 2):
            support = targets[:, c]

        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            query = torch.arange(start, end, device=device)
            sims = torch.matmul(feats_c[query], feats_c.T)
            actual_k = min(k + 1, sims.size(1))
            topk = sims.topk(actual_k, dim=1).indices
            neigh = topk[:, 1:] if actual_k > 1 else topk
            if neigh.numel() == 0:
                purity = torch.zeros(len(query), dtype=torch.float32, device=device)
            else:
                purity = support[neigh].float().mean(dim=1)
            purity_scores[query, c] = purity

        pos_purity = purity_scores[pos, c]
        is_clean = pos_purity >= threshold
        knn_clean[pos[is_clean], c] = True
        knn_noisy[pos[~is_clean], c] = True

    return knn_clean, knn_noisy, purity_scores


def build_dynamic_hard_clean_mask(loss_risk_mask, fkl_mask, ema_preds, ema_vars, proto_clean, knn_clean, ema_thresh=0.5, var_thresh=0.2):
    dynamic_support = fkl_mask.bool() & (ema_preds >= ema_thresh) & (ema_vars <= var_thresh)
    geometry_support = proto_clean.bool() & knn_clean.bool()
    return loss_risk_mask.bool() & (dynamic_support | geometry_support)


def build_label_reliability(
    targets,
    global_label_mask,
    loss_clean_mask,
    loss_risk_mask,
    mee_easy_noisy_mask,
    dynamic_hard_clean_mask,
    proto_clean_mask,
    proto_noisy_mask,
    knn_clean_mask,
    knn_noisy_mask,
    fkl_mask,
):
    targets = targets.bool()
    positive = targets
    clean_score = (
        0.6 * loss_clean_mask.float()
        + 0.8 * global_label_mask.float()
        + 0.6 * fkl_mask.float()
        + 1.0 * dynamic_hard_clean_mask.float()
        + 0.5 * proto_clean_mask.float()
        + 0.5 * knn_clean_mask.float()
    )
    noisy_score = (
        0.7 * loss_risk_mask.float()
        + 1.2 * mee_easy_noisy_mask.float()
        + 0.5 * proto_noisy_mask.float()
        + 0.5 * knn_noisy_mask.float()
        + 0.5 * (positive & (~global_label_mask.bool())).float()
    )
    reliability = torch.where(
        positive,
        torch.sigmoid(clean_score - noisy_score),
        torch.ones_like(clean_score),
    )

    hard_clean_mask = positive & (
        dynamic_hard_clean_mask.bool()
        | (loss_clean_mask.bool() & (~mee_easy_noisy_mask.bool()))
        | (proto_clean_mask.bool() & knn_clean_mask.bool())
    )
    clear_noisy_fp = positive & (
        mee_easy_noisy_mask.bool()
        | ((loss_risk_mask.bool() | (~global_label_mask.bool())) & proto_noisy_mask.bool() & knn_noisy_mask.bool())
    ) & (~hard_clean_mask)
    fp_mask = positive & (~hard_clean_mask) & (
        clear_noisy_fp
        | (~global_label_mask.bool())
        | (reliability < 0.5)
    )

    return reliability, fp_mask, hard_clean_mask, clear_noisy_fp


def mine_false_negative_mask(
    targets,
    ema_preds,
    ema_vars,
    proto_sim,
    knn_purity,
    cond_prob_matrix,
    ema_thresh=0.7,
    var_thresh=0.2,
    proto_sim_thresh=0.35,
    knn_purity_thresh=0.05,
    prior_thresh=0.2,
):
    targets = targets.bool()
    fn_mask = torch.zeros_like(targets, dtype=torch.bool)
    num_samples, num_classes = targets.shape

    for c in range(num_classes):
        candidate = (
            (~targets[:, c])
            & (ema_preds[:, c] >= ema_thresh)
            & (ema_vars[:, c] <= var_thresh)
        )
        if proto_sim is not None:
            candidate = candidate & (proto_sim[:, c] >= proto_sim_thresh)
        if knn_purity is not None:
            candidate = candidate & (knn_purity[:, c] >= knn_purity_thresh)
        if not candidate.any():
            continue

        related = torch.where(cond_prob_matrix[:, c] > prior_thresh)[0]
        if len(related) > 1:
            support_scores = torch.zeros(num_samples, dtype=torch.float32, device=targets.device)
            for rc in related:
                if int(rc) != c:
                    support_scores += targets[:, rc].float()
            candidate = candidate & (support_scores > 0)
        fn_mask[:, c] = candidate

    return fn_mask


def generate_asymmetric_soft_targets(targets, ema_preds, label_reliability, fp_mask, clear_noisy_fp, hard_clean_mask, fn_mask, fp_soft_max=0.5):
    final_targets = targets.float().clone()
    positive = targets.bool()

    uncertain_positive = positive & (~hard_clean_mask.bool()) & ((label_reliability < 0.75) | fp_mask.bool())
    blended = label_reliability * 1.0 + (1.0 - label_reliability) * ema_preds
    final_targets[uncertain_positive] = blended[uncertain_positive]

    final_targets[clear_noisy_fp] = torch.clamp(ema_preds[clear_noisy_fp], max=fp_soft_max)
    final_targets[fn_mask] = ema_preds[fn_mask]
    return final_targets.clamp_(0.0, 1.0)


def save_reliability_outputs(output_dir, tensors):
    os.makedirs(output_dir, exist_ok=True)
    for name, tensor in tensors.items():
        torch.save(tensor.detach().cpu(), os.path.join(output_dir, f'{name}.pt'))


def save_reliability_summary(output_dir, targets, tensors):
    os.makedirs(output_dir, exist_ok=True)
    targets_cpu = targets.detach().cpu().bool()
    rows = []
    for c in range(targets_cpu.size(1)):
        row = {'class': c, 'positives': int(targets_cpu[:, c].sum())}
        for name, tensor in tensors.items():
            if tensor.dim() == 2 and tensor.size(1) == targets_cpu.size(1):
                tensor_cpu = tensor.detach().cpu()
                if tensor_cpu.dtype == torch.bool or name.endswith('_mask'):
                    row[name] = int(tensor_cpu.bool()[:, c].sum())
                else:
                    pos = targets_cpu[:, c]
                    value = tensor_cpu[pos, c].float().mean().item() if pos.any() else 0.0
                    row[f'{name}_mean_pos'] = round(value, 6)
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(os.path.join(output_dir, 'stage1_reliability_summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
