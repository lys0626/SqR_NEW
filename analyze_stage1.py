import os
import torch
import argparse
from torch.utils.data import DataLoader
from lib.dataset.get_dataset import get_datasets

def parser_args():
    parser = argparse.ArgumentParser(description='Analyze Stage 1 Outputs (Clean/Noisy Stats)')
    # == 原有基础参数 (与 stage1_main.py 完全一致) ==
    parser.add_argument('--dataname', default='mimic', choices=['coco14', 'mimic', 'nih'])
    parser.add_argument('--dataset_dir', default='/data/mimic_cxr/PA/7_1_2', type=str)
    parser.add_argument('--output', default='/data/dsj/lys/SqR-NEW/experiment/ASL_0.5_test_0.95_OneCycle/mimic/stage1_splicemix-cl_q2l', metavar='DIR', help='path to output folder')
    parser.add_argument('--num_class', default=13, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'SGD'])
    parser.add_argument('--scheduler', default='OneCycle', type=str)
    parser.add_argument('--step_size', default=40, type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    
    parser.add_argument('--eps', default=1e-5, type=float)           
    parser.add_argument('--gamma_pos', default=0, type=float)
    parser.add_argument('--gamma_neg', default=4, type=float)
    parser.add_argument('--loss_clip', default=0.05, type=float)  

    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--val_interval', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--seed', default=95, type=int)

    # == Q2L Transformer 参数 ==
    parser.add_argument('--enc_layers', default=1, type=int)
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--nheads', default=4, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--keep_other_self_attn_dec', action='store_true')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true')
    parser.add_argument('--keep_input_proj', action='store_true')
    
    # ================= 标签级 MEE 核心对齐参数 =================
    parser.add_argument('--warm_up_epochs', default=6, type=int)
    parser.add_argument('--fkl_consecutive_epochs', default=5, type=int)
    parser.add_argument('--early_cutting_rate', default=1.5, type=float)
    parser.add_argument('--newremove_rate', default=90000, type=int)
    parser.add_argument('--top_conf_ratio', default=0.2, type=float)
    parser.add_argument('--low_grad_ratio', default=0.2, type=float)
    
    # 单卡设备及归一化
    parser.add_argument('-cd', '--cuda_devices', default=[0], nargs='+', type=int)
    parser.add_argument('--orid_norm', action='store_true', default=False)

    # Phase 控制参数
    parser.add_argument("--i_rate_1", type=int, default=3)
    parser.add_argument("--i_rate_2", type=int, default=3)
    parser.add_argument("--i_rate_3", type=int, default=3)
    parser.add_argument("--i_rate_4", type=int, default=0)
    
    # 动态保留比例
    parser.add_argument("--remove_rate_1", type=float, default=0.99)
    parser.add_argument("--remove_rate_2", type=float, default=0.99)
    parser.add_argument("--remove_rate_3", type=float, default=0.99)
    parser.add_argument("--remove_rate_4", type=float, default=0.99)
    
    args = parser.parse_args()
    return args

def main():
    args = parser_args()
    
    print(f"正在加载 {args.dataname} 训练数据集以获取全局标签...")
    train_dataset, _ = get_datasets(args)
    loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=args.workers)
    
    all_targets = torch.zeros((len(train_dataset), args.num_class), dtype=torch.bool)
    print("正在扫描全局标签，请稍候...")
    for _, targets, indices in loader:
        all_targets[indices] = (targets == 1)
        
    # =============== 任务 1：统计训练集总计标注为1的各个标签数量 ===============
    total_positives = all_targets.sum(dim=0)
    print("\n" + "="*50)
    print("1. 训练集总计标注为1的各个标签数量:")
    print("="*50)
    for i in range(args.num_class):
        print(f"类别 {i}: {total_positives[i].item()} 个")

    # =============== 读取 Stage 1 产物 ===============
    clean_path = os.path.join(args.output, 'clean_indices.pt')
    noisy_path = os.path.join(args.output, 'noisy_indices.pt')
    dict_path = os.path.join(args.output, 'noise_clean_labels_dict.pt')
    
    if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
        print(f"\n[错误] 未在 {args.output} 找到相关 .pt 文件。")
        return
        
    clean_indices = torch.load(clean_path)
    noisy_indices = torch.load(noisy_path)
    
    print(f"\n>>> 成功加载 Stage 1 结果: 干净样本数 = {len(clean_indices)}, 噪声样本数 = {len(noisy_indices)}")

    # =============== 任务 2：统计 clean_indices 标注为1的各个标签数量 ===============
    clean_targets = all_targets[clean_indices]
    clean_positives = clean_targets.sum(dim=0)
    print("\n" + "="*50)
    print("2. clean_indices 中各标签标注为1的数量 (样本本身判定为干净):")
    print("="*50)
    for i in range(args.num_class):
        ratio = (clean_positives[i].item() / (total_positives[i].item() + 1e-8)) * 100
        print(f"类别 {i}: {clean_positives[i].item()} 个 (保留率: {ratio:.2f}%)")

    # =============== 任务 3：统计 noisy_indices 包含的正标签数量 ===============
    noisy_targets = all_targets[noisy_indices]
    noisy_positives = noisy_targets.sum(dim=0)
    print("\n" + "="*50)
    print("3. noisy_indices 各类别分布情况 (样本被流放，包含的正标签总数):")
    print("="*50)
    for i in range(args.num_class):
        noise_ratio = (noisy_positives[i].item() / (total_positives[i].item() + 1e-8)) * 100
        print(f"类别 {i}: {noisy_positives[i].item()} 个 (占比: {noise_ratio:.2f}%)")

    # =============== 任务 4：统计 noise_clean_labels_dict 中的真正干净标签 ===============
    if os.path.exists(dict_path):
        noise_clean_labels_dict = torch.load(dict_path)
        rescued_counts = torch.zeros(args.num_class, dtype=torch.int32)
        
        # 遍历字典：{ noisy_sample_idx : [clean_label_1, clean_label_2, ...] }
        for n_idx, clean_lbls in noise_clean_labels_dict.items():
            for lbl in clean_lbls:
                if all_targets[n_idx, lbl] == True:
                    rescued_counts[lbl] += 1
                
        print("\n" + "="*50)
        print("4. noise_clean_labels_dict 中各类别真正干净的标签数量 (从噪声样本中抢救回来的标签):")
        print("="*50)
        for i in range(args.num_class):
            rescued = rescued_counts[i].item()
            total_in_noise = noisy_positives[i].item()
            # 计算在被流放的该类标签中，有多少比例是真正干净的
            rescue_ratio = (rescued / (total_in_noise + 1e-8)) * 100
            print(f"类别 {i}: {rescued} 个 (占该类被流放标签的: {rescue_ratio:.2f}%)")
            
        print("\n" + "="*50)
        print("★ 最终 Stage 2 可用该类别的真实正标签总数 = (任务2) + (任务4)")
        print("="*50)
        for i in range(args.num_class):
            final_usable = clean_positives[i].item() + rescued_counts[i].item()
            final_ratio = (final_usable / (total_positives[i].item() + 1e-8)) * 100
            print(f"类别 {i}: {final_usable} 个 (整体利用率: {final_ratio:.2f}%)")
            
    else:
        print(f"\n[提示] 未找到 {dict_path}，跳过字典统计。")

if __name__ == '__main__':
    main()