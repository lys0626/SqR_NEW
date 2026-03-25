import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lib.models.query2label import build_q2l
from lib.dataset.get_dataset import get_datasets

def parse_args():
    parser = argparse.ArgumentParser(description='Offline CAM 2x2 Mask Generation')
    parser.add_argument('--dataname', default='mimic', choices=['mimic', 'nih'])
    parser.add_argument('--dataset_dir', default='/data/mimic_cxr/PA/7_1_2')
    parser.add_argument('--output', metavar='DIR', help='path to Stage 1 output folder (e.g. ./experiment/ASL/...)')
    parser.add_argument('--num_class', default=13, type=int)
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--img_size', default=448, type=int)
    
    # 核心模型参数 (保持与 Stage 1 一致)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--enc_layers', default=1, type=int)
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--nheads', default=4, type=int)
    parser.add_argument('--keep_input_proj', action='store_true', default=True)
    parser.add_argument('--keep_other_self_attn_dec', action='store_true')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true')
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--orid_norm', action='store_true', default=False)
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"==> Loading Stage 1 Best Model from: {args.output}")
    model = build_q2l(args)
    checkpoint_path = os.path.join(args.output, 'model_best.pth.tar')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to(device)
    model.eval()

    print("==> Loading Data Indices and Labels Dictionary...")
    noisy_indices_path = os.path.join(args.output, 'noisy_indices.pt')
    dict_path = os.path.join(args.output, 'noise_clean_labels_dict.pt')
    
    noisy_indices_set = set(torch.load(noisy_indices_path))
    noise_clean_labels_dict = torch.load(dict_path)

    print("==> Initializing DataLoader (Shuffle=False)...")
    train_dataset, _ = get_datasets(args)
    # 重点：必须 shuffle=False
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    noise_cam_masks = {}
    
    threshold = 0.20 # 因为 4 个格子和为 1，平均是 0.25。0.20 是一个安全阈值，确保抓住主体病灶

    with torch.no_grad():
        for images, targets, indices in tqdm(train_loader, desc="Extracting CAM 2x2 Masks"):
            images = images.to(device)
            
            # 开启 return_attn
            _, _, _, src, attn_weights = model(images, return_attn=True)
            
            # src 维度 [Batch, 2048, H, W] -> 通常 448 输入对应 H=14, W=14
            H, W = src.shape[-2], src.shape[-1]
            
            # 遍历 Batch 内部
            for b in range(images.size(0)):
                global_idx = indices[b].item()
                
                # 如果是干净样本，直接跳过，它不需要 CAM
                if global_idx not in noisy_indices_set:
                    continue
                    
                noise_cam_masks[global_idx] = {}
                clean_classes = noise_clean_labels_dict.get(global_idx, [])
                
                for c in clean_classes:
                    # 获取该样本、该类的注意力图 [H*W]
                    # PyTorch MultiheadAttention 返回的维度通常为 [Batch, Num_Queries, Source_Len]
                    cls_attn = attn_weights[b, c, :] 
                    
                    # 恢复为二维空间 [1, 1, H, W]
                    cls_attn_2d = cls_attn.view(1, 1, H, W)
                    
                    # 极速自适应池化到 2x2
                    grid_energy = F.adaptive_avg_pool2d(cls_attn_2d, (2, 2)).squeeze() # [2, 2]
                    
                    # 归一化使得 4 个格子概率和为 1
                    grid_energy = grid_energy / (grid_energy.sum() + 1e-8)
                    
                    # 布尔掩码判定 (True 代表有病灶，需要抠图)
                    mask_2x2 = grid_energy >= threshold
                    
                    # 存入字典 (转移到 CPU 节省内存)
                    noise_cam_masks[global_idx][c] = mask_2x2.cpu()

    save_path = os.path.join(args.output, 'noise_cam_masks.pt')
    torch.save(noise_cam_masks, save_path)
    print(f"\n==> Successfully saved CAM masks for {len(noise_cam_masks)} noisy images to: {save_path}")

if __name__ == '__main__':
    main()