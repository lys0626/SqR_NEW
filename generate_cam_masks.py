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
    parser.add_argument('--output', metavar='DIR', help='path to Stage 1 output folder')
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
    
    # 【修复 1】：补齐缺失的 pretrained 参数，防止 build_backbone 崩溃
    parser.add_argument('--pretrained', action='store_true', default=False)
    # ================= 【修复：底层 Dataset 所需的兼容占位参数】 =================
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=False, help='evaluate model on validation set')
    parser.add_argument('--cutout', action='store_true', default=False, help='Use cutout')
    parser.add_argument('--n_holes', type=int, default=1, help='number of holes for cutout')
    parser.add_argument('--length', type=int, default=16, help='length of holes for cutout')
    # =========================================================================
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
    threshold = 0.20 # 抓取主体病灶的安全阈值

    with torch.no_grad():
        for images, targets, indices in tqdm(train_loader, desc="Extracting CAM 2x2 Masks"):
            images = images.to(device)
            
            # 开启 return_attn
            _, _, _, src, attn_weights = model(images, return_attn=True)
            H, W = src.shape[-2], src.shape[-1]
            
            # 遍历 Batch 内部
            for b in range(images.size(0)):
                global_idx = indices[b].item()
                
                # 如果是干净样本，直接跳过
                if global_idx not in noisy_indices_set:
                    continue
                    
                # 【修复 2】：解析 Stage 1 传来的纯净硬标签 Tensor
                clean_labels_tensor = noise_clean_labels_dict.get(global_idx, None)
                if clean_labels_tensor is None:
                    continue
                
                # 获取被判定为阳性的真实类别索引 (例如: [0, 3] 代表 Infiltration 和 Nodule)
                clean_class_indices = torch.where(clean_labels_tensor > 0.5)[0]
                
                if len(clean_class_indices) == 0:
                    # 如果提纯后没有任何阳性标签，赋予一个默认的全 False 掩码
                    noise_cam_masks[global_idx] = torch.zeros((2, 2), dtype=torch.bool)
                    continue

                # 【修复 3】：初始化一个 2x2 全局掩码，对多种疾病的病灶取并集
                combined_mask = torch.zeros((2, 2), dtype=torch.bool)
                
                for c in clean_class_indices:
                    cls_idx = c.item()
                    cls_attn = attn_weights[b, cls_idx, :] 
                    
                    # 恢复为二维空间
                    cls_attn_2d = cls_attn.view(1, 1, H, W)
                    
                    # 极速自适应池化到 2x2
                    grid_energy = F.adaptive_avg_pool2d(cls_attn_2d, (2, 2)).squeeze() 
                    
                    # 归一化使得 4 个格子概率和为 1
                    grid_energy = grid_energy / (grid_energy.sum() + 1e-8)
                    
                    # 布尔掩码判定
                    mask_2x2 = grid_energy >= threshold
                    
                    # 取并集 (OR)：只要任意一个疾病在这里有病灶，这个格子就可以用来抠图！
                    combined_mask = combined_mask | mask_2x2.cpu()
                    
                noise_cam_masks[global_idx] = combined_mask

    save_path = os.path.join(args.output, 'noise_cam_masks.pt')
    torch.save(noise_cam_masks, save_path)
    print(f"\n==> Successfully saved CAM masks for {len(noise_cam_masks)} noisy images to: {save_path}")

if __name__ == '__main__':
    main()