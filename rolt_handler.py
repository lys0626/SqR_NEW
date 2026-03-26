import torch
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def plot_feature_tsne(feats, targets, class_idx, epoch, class_names=None, save_dir="./diagnostics"):
    """
    [诊断工具 1 - 改进版] 用 t-SNE 可视化特定类别的正负样本特征分布，直接标注真实类别名称
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取类别名称，如果未提供列表则退化为 Class X
    if class_names is not None and len(class_names) > class_idx:
        class_name = class_names[class_idx]
    else:
        class_name = f"Class {class_idx}"
        
    # 1. 获取正样本和负样本的全局索引
    pos_indices = torch.where(targets[:, class_idx] == 1)[0]
    neg_indices = torch.where(targets[:, class_idx] == 0)[0]
    
    num_pos = len(pos_indices)
    if num_pos < 10: 
        print(f"[Diagnostics] {class_name} has too few positive samples ({num_pos}). Skipping t-SNE.")
        return
        
    # 2. 随机采样负样本 (1:3 比例)
    perm = torch.randperm(len(neg_indices))
    num_neg_to_sample = min(num_pos * 3, len(neg_indices))
    sampled_neg_indices = neg_indices[perm[:num_neg_to_sample]]
    
    # 3. 提取特征并转为 numpy
    pos_feats = feats[pos_indices, class_idx, :].cpu().numpy()
    neg_feats = feats[sampled_neg_indices, class_idx, :].cpu().numpy()
    
    all_feats = np.vstack((pos_feats, neg_feats))
    labels = np.concatenate((np.ones(num_pos), np.zeros(num_neg_to_sample)))
    
    # 4. 运行 t-SNE
    print(f"[Diagnostics] Running t-SNE for '{class_name}' at Epoch {epoch}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    feats_2d = tsne.fit_transform(all_feats)
    
    # 5. 绘图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=feats_2d[:, 0], y=feats_2d[:, 1], hue=labels, 
                    palette={1.0: 'red', 0.0: 'royalblue'}, 
                    alpha=0.6, s=30)
    
    # 标题直接使用疾病名称
    plt.title(f't-SNE Feature Representation - {class_name} (Epoch {epoch})', fontsize=14, fontweight='bold')
    
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ['Negative (0)', 'Positive (1)'], title='Ground Truth')
    
    # 处理文件名：将疾病名称中的空格或斜杠替换为下划线，避免保存报错
    safe_class_name = class_name.replace(" ", "_").replace("/", "_")
    save_path = os.path.join(save_dir, f'tsne_{safe_class_name}_ep{epoch}.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Diagnostics] t-SNE plot saved to {save_path}")
def plot_query_similarity(model, epoch, save_dir="./diagnostics"):
    """
    [诊断工具 2] 计算并绘制 Q2L Query Embeddings 的余弦相似度矩阵
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 兼容 DDP 和单卡，安全获取原始模型
    real_model = model.module if hasattr(model, 'module') else model
    
    # 2. 定位 Query Embeddings (Q2L官方实现中通常命名为 query_embed)
    try:
        if hasattr(real_model, 'query_embed'):
            query_weights = real_model.query_embed.weight.detach() # [num_classes, hidden_dim]
        elif hasattr(real_model, 'transformer') and hasattr(real_model.transformer, 'query_embed'):
            query_weights = real_model.transformer.query_embed.weight.detach()
        else:
            print("[Diagnostics] Could not locate 'query_embed' in model. Skipping similarity matrix.")
            return
    except Exception as e:
        print(f"[Diagnostics] Error fetching query embedding: {e}")
        return
        
    # 3. 计算两两之间的余弦相似度
    query_norm = torch.nn.functional.normalize(query_weights, p=2, dim=1)
    sim_matrix = torch.matmul(query_norm, query_norm.T).cpu().numpy()
    
    # 4. 绘制热力图 
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="Reds", vmin=0.0, vmax=1.0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title(f'Query Embedding Cosine Similarity Matrix (Epoch {epoch})\n(High off-diagonal values indicate Query Collapse)', pad=20)
    plt.xlabel('Class Index')
    plt.ylabel('Class Index')
    
    save_path = os.path.join(save_dir, f'query_sim_ep{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Diagnostics] Query similarity matrix saved to {save_path}")
def plot_knn_purity(purities, class_idx, epoch, threshold, save_dir):
    """
    可视化 KNN 局部邻居的纯度 (Purity) 分布
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    
    # 绘制纯度直方图 (取值范围 0 到 1)
    plt.hist(purities, bins=20, range=(0, 1), density=False, alpha=0.7, color='teal', edgecolor='black')
    
    # 画出判定阈值线
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
    
    plt.title(f'KNN Local Purity for Class {class_idx} at Epoch {epoch}\n(Higher Purity = Cleaner Lesion)')
    plt.xlabel('Proportion of Positive Neighbors (Local Purity)')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = os.path.join(save_dir, f'knn_purity_ep{epoch}_cls{class_idx}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
def plot_distance_gmm_fit(dists, gmm, class_idx, epoch, save_dir="./gmm_distance_plots"):
    """
    可视化基于 Cosine Distance 的 1D GMM 拟合效果
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
        
    # 1. 绘制经验距离的直方图
    plt.hist(dists, bins=50, density=True, alpha=0.5, color='gray', label='Empirical Cosine Distance')
        
    # 2. 生成 X 轴的平滑点 (余弦距离范围通常在 0 到 2 之间，大部分集中在 0 到 1)
    x = np.linspace(dists.min(), dists.max(), 1000).reshape(-1, 1)
        
    # 3. 提取 GMM 参数
    weights = gmm.weights_
    means = gmm.means_
    covars = gmm.covariances_
    
    # 4. 计算 PDF
    pdf0 = weights[0] * norm.pdf(x, means[0, 0], np.sqrt(covars[0, 0, 0]))
    pdf1 = weights[1] * norm.pdf(x, means[1, 0], np.sqrt(covars[1, 0, 0]))
    
    # 5. 绘制单组件曲线 (均值小的靠近左侧，代表 Clean 样本)
    plt.plot(x, pdf0, color='blue', linewidth=2, label=f'Comp 0 (Mean={means[0,0]:.3f})')
    plt.plot(x, pdf1, color='red', linewidth=2, label=f'Comp 1 (Mean={means[1,0]:.3f})')
    
    # 6. 绘制总和曲线
    plt.plot(x, pdf0 + pdf1, color='black', linestyle='--', linewidth=2, label='GMM Sum')
    
    # 7. 寻找判别边界并画绿线
    diff = np.abs(pdf0 - pdf1)
    valid_idx = np.where((x > means.min()) & (x < means.max()))[0]
    if len(valid_idx) > 0:
        intersection_idx = valid_idx[np.argmin(diff[valid_idx])]
        decision_boundary = x[intersection_idx][0]
        plt.axvline(x=decision_boundary, color='green', linestyle=':', linewidth=2, label=f'Boundary={decision_boundary:.3f}')
        
    plt.title(f'Distance-based GMM Fit for Class {class_idx} at Epoch {epoch}')
    plt.xlabel('Cosine Distance (1.0 - Cosine Similarity)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
        
    # 保存图片
    plt.savefig(os.path.join(save_dir, f'gmm_dist_ep{epoch}_cls{class_idx}.png'), dpi=150)
    plt.close()
class RoLT_Handler(object):
    def __init__(self, args, model, train_loader, num_classes, feature_dim):
        """
        RoLT 逻辑处理器
        :param args: 参数配置
        :param model: Q2L 模型
        :param train_loader: 训练数据加载器 (必须返回 index)
        :param num_classes: 类别数 (MIMIC=13, NIH=14)
        :param feature_dim: Transformer Decoder 输出维度 (如 2048)
        """
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化类原型 (Centroids) [Num_Classes, Feature_Dim]
        self.prototypes = torch.zeros(self.num_classes, self.feature_dim).to(self.device)
        
        # 存储上一轮的 clean mask 以便 momentum 更新 (可选，此处暂未实现以保持简单)
        self.prob_history = torch.zeros(len(train_loader.dataset)).to(self.device)
    def step(self, epoch):
        print(f"\n==> [RoLT] Epoch {epoch}: Starting Logic Flow...")
        
        # 1. 提取完整特征 (注意：这里必须是你之前修改过的、没有 mask 归零的特征！)
        all_feats, all_logits, all_targets, all_indices = self.extract_features_and_mask()
        # ========================================================
        # [特征体检] 插入特征诊断代码 (例如：每 5 个 Epoch 做一次体检)
        # ========================================================
        try:
            diag_dir = os.path.join(self.args.output, "diagnostics")
        except AttributeError:
            diag_dir = "./diagnostics"
            
        if epoch==15:  
            # 诊断一：Query 是否坍缩？
            plot_query_similarity(self.model, epoch, save_dir=diag_dir)
            
            # 诊断二：查看前 3 个疾病类别的特征分布 (你可以修改 class_idx 查看特定疾病)
            
            for c in range(self.num_classes):
                plot_feature_tsne(all_feats, all_targets, class_idx=c, epoch=epoch,save_dir=diag_dir)
        # ========================================================
        # 2. 计算初始类原型
        self.update_prototypes(all_feats, all_targets)
        
        # 3. [重磅替换] 采用 KNN 清洗代替 GMM
        # k=50: 看最近的 50 个片子
        # purity_threshold=0.2: 只要这 50 个片子里有 10 个（20%）也标了有病，就认定是干净的！
        label_clean_matrix = self.knn_cleaning(all_feats, all_targets, epoch, k=50, default_purity=0.1)
        
        # 4. 样本级筛选计算 (你之前指出的 clean_counts 逻辑)
        clean_mask_dict = self.apply_sample_selection_rule(label_clean_matrix, all_targets, all_indices)
        
        # 5. 二次更新原型 (仅用绝对干净的样本)
        self.refine_prototypes(all_feats, all_targets, all_indices, clean_mask_dict)
        
        # 6. 生成软伪标签 (包含我们之前优化的 NCM/ERM 双重校验和温和缓冲)
        soft_label_dict = self.generate_soft_labels(all_feats, all_logits, all_targets, all_indices, clean_mask_dict, label_clean_matrix)       
        
        print(f"==> [RoLT] Epoch {epoch}: Logic Flow Finished.\n")
        return clean_mask_dict, soft_label_dict
    def step_center(self, epoch):
        """
        每个 Epoch 开始时调用。
        执行：特征提取 -> Masking -> 原型计算 -> GMM 清洗 -> 样本筛选 -> 软标签生成
        :return: (clean_mask_dict, soft_label_dict)
        """
        print(f"\n==> [RoLT] Epoch {epoch}: Starting Logic Flow...")
        
        # 1. 冻结模型 & 提取特征
        all_feats, all_logits, all_targets, all_indices = self.extract_features_and_mask()
        
        # 2. 计算初始类原型 (使用当前 Epoch 所有正样本)
        self.update_prototypes(all_feats, all_targets)
        
        # 3. GMM 清洗 (Per-Class)
        # 返回: [N_samples, N_classes] 的 Boolean 矩阵，表示该样本在该标签上是否“干净”
        # 注意：仅针对 targets=1 的标签进行判定，targets=0 的默认为 True (或忽略)
        label_clean_matrix = self.gmm_cleaning_center(all_feats, all_targets,epoch)
        
        # 4. 样本级筛选 (+2 规则)
        # 判定哪些样本有资格进入 "SpliceMix" (即 Clean 样本)
        # 返回: {sample_index: is_clean_sample(bool)}
        clean_mask_dict = self.apply_sample_selection_rule(label_clean_matrix, all_targets, all_indices)
        
        # 5. 基于筛选出的“值得信赖”样本，二次更新原型 (Refinement)
        self.refine_prototypes(all_feats, all_targets, all_indices, clean_mask_dict)
        
        # 6. 生成软伪标签 (Soft Pseudo Labels)
        # 为“噪声样本”生成用于校正的软标签
        # 返回: {sample_index: soft_targets(Tensor)}
        soft_label_dict = self.generate_soft_labels(all_feats, all_logits, all_targets, all_indices, clean_mask_dict, label_clean_matrix)       
        print(f"==> [RoLT] Epoch {epoch}: Logic Flow Finished.\n")
        
        return clean_mask_dict, soft_label_dict
    def step_loss(self, epoch):
        print(f"\n==> [RoLT] Epoch {epoch}: Starting Logic Flow (Small-Loss Criterion)...")
        
        # 1. 冻结模型 & 提取特征与 Logits
        all_feats, all_logits, all_targets, all_indices = self.extract_features_and_mask()
        
        # 2. 计算初始类原型 (用于后续计算软标签中的 p_ncm)
        self.update_prototypes(all_feats, all_targets)
        
        # ================== 新增: 计算 Loss 矩阵 ==================
        # 计算所有样本在每个标签上的独立 BCE Loss
        # reduction='none' 保证输出形状与 all_logits 一致，即 [N, C]
        all_losses = F.binary_cross_entropy_with_logits(
            all_logits, 
            all_targets.float(), 
            reduction='none'
        )
        # ==========================================================
        
        # 3. GMM 清洗 (传入 all_losses 代替 all_feats)
        label_clean_matrix = self.gmm_cleaning_loss(all_losses, all_targets)
        
        # 4. 样本级筛选
        clean_mask_dict = self.apply_sample_selection_rule(label_clean_matrix, all_targets, all_indices)
        
        # 5. 二次更新原型
        self.refine_prototypes(all_feats, all_targets, all_indices, clean_mask_dict)
        
        # 6. 生成软伪标签
        soft_label_dict = self.generate_soft_labels(all_feats, all_logits, all_targets, all_indices, clean_mask_dict, label_clean_matrix)        
        
        print(f"==> [RoLT] Epoch {epoch}: Logic Flow Finished.\n")
        return clean_mask_dict, soft_label_dict
    def extract_features_and_mask(self):
        """
        遍历数据集，提取特征，并根据 Target 进行 Masking
        Masking 规则: 标签对应的 Decoder 输出保留，不属于该标签的置为 0
        """
        self.model.eval()
        # 确保冻结
        for param in self.model.parameters():
            param.requires_grad = False
            
        all_feats = []
        all_logits = [] # [新增] 存储 Logits
        all_targets = []
        all_indices = []
        
        with torch.no_grad():
            for i, (images, targets, indices) in tqdm(enumerate(self.train_loader), 
                                                     total=len(self.train_loader), 
                                                     desc="[RoLT] Extracting Features"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Q2L Forward: 返回 (logits, last_layer_feats)
                # last_layer_feats shape: [Batch, Num_Classes, Feature_Dim]
                logits, features, _,_ = self.model(images)
                
                # --- Masking Logic ---
                # 仅保留 Target=1 对应的 Query Slot 特征
                # targets shape: [B, C] -> mask shape: [B, C, 1]
                # mask = targets.unsqueeze(-1)
                # masked_features = features * mask 
                masked_features = features
                all_feats.append(masked_features.cpu())   # 存入 CPU 避免显存爆炸
                all_logits.append(logits.cpu()) # [新增]
                all_targets.append(targets.cpu())
                all_indices.append(indices.cpu())
                
        # 拼接
        all_feats = torch.cat(all_feats, dim=0).to(self.device)     # [N, C, D]
        all_logits = torch.cat(all_logits, dim=0).to(self.device) # [新增]
        all_targets = torch.cat(all_targets, dim=0).to(self.device) # [N, C]
        all_indices = torch.cat(all_indices, dim=0).to(self.device) # [N]
        
        return all_feats, all_logits, all_targets, all_indices # [修改]

    def update_prototypes(self, feats, targets):
        """
        计算类原型 (所有正样本的特征均值)
        """
        print("[RoLT] Updating Prototypes...")
        new_prototypes = torch.zeros_like(self.prototypes)
        
        for c in range(self.num_classes):
            # 获取该类标签为 1 的样本索引
            pos_indices = (targets[:, c] == 1)
            
            if pos_indices.sum() > 0:
                # 取出对应的特征向量 (已经是 masked 的，非 0)
                class_feats = feats[pos_indices, c, :] 
                # L2 归一化 (RoLT 中常用的操作，防止模长影响距离)
                class_feats = F.normalize(class_feats, p=2, dim=1)
                
                # 计算均值并归一化更新
                proto = class_feats.mean(dim=0)
                new_prototypes[c] = F.normalize(proto, p=2, dim=0)
            else:
                # 如果该类在本 Epoch 没有正样本 (极少见)，保持旧原型
                new_prototypes[c] = self.prototypes[c]
                
        self.prototypes = new_prototypes
    def knn_cleaning(self, feats, targets, epoch, k=50, default_purity=0.1):
        """
        [阶梯式自适应架构] 基于 KNN 局部近邻的噪声清洗
        针对长尾医疗数据，采用极其温柔的阶梯阈值保护罕见病灶。
        """
        print(f"\n[RoLT] Running Adaptive KNN Cleaning (K={k}, Default Thresh={default_purity})...")
        label_clean_matrix = torch.zeros_like(targets, dtype=torch.bool).to(self.device)
        
        # 准备可视化路径
        try:
            base_out_dir = self.args.output 
        except AttributeError:
            base_out_dir = "./experiment/robust_run"
        vis_dir = os.path.join(base_out_dir, "knn_visualizations")

        for c in range(self.num_classes):
            #计算在标签中每个类别的正样本数量
            pos_indices = torch.where(targets[:, c] == 1)[0]
            num_samples = len(pos_indices)
            
            # ================= 阶梯式保护与阈值分配 =================
            # 1. 绝对保护区 (样本 < 1000)：极其罕见，直接无条件信任
            if num_samples < 100: 
                print(f"  -> Class {c:02d} | Samples: {num_samples:<5} | Strategy: [Bypass] Absolute Protection")
                label_clean_matrix[pos_indices, c] = True
                continue
                
            # 2. 弱势群体区 (1000 <= 样本 < 2000)：大幅放宽标准 (0.05)
            elif num_samples < 200:
                current_thresh = 0.05
                print(f"  -> Class {c:02d} | Samples: {num_samples:<5} | Strategy: [Relaxed] Threshold = {current_thresh}")
            
            # 3. 头部大类区 (样本 >= 2000)：采用默认标准 (0.1)
            else:
                current_thresh = default_purity
                print(f"  -> Class {c:02d} | Samples: {num_samples:<5} | Strategy: [Normal]  Threshold = {current_thresh}")
            # ========================================================
                
            # 提取所有样本的类别c的特征并归一化
            class_feats = feats[:, c, :] 
            class_feats = F.normalize(class_feats, p=2, dim=1)
            
            chunk_size = 2048
            all_purities = []
            
            for start_idx in range(0, num_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, num_samples)
                batch_pos_feats = class_feats[pos_indices[start_idx:end_idx]] 
                
                sims = torch.matmul(batch_pos_feats, class_feats.T)
                
                actual_k = min(k + 1, sims.size(1))
                #获取sims中每行最大的k+1个值的索引（包括自己），因为第一个是自己，所以取后k个作为邻居
                _, topk_indices = sims.topk(actual_k, dim=1)
                
                neighbor_indices = topk_indices[:, 1:]
                neighbor_targets = targets[neighbor_indices, c] 
                
                purity = neighbor_targets.float().mean(dim=1) 
                all_purities.append(purity)
                
                # 【应用当前类别的动态阈值】
                is_clean = (purity >= current_thresh)
                
                real_indices = pos_indices[start_idx:end_idx]
                clean_real_indices = real_indices[is_clean]
                label_clean_matrix[clean_real_indices, c] = True
                
            # --- 动态可视化 ---
            target_epoch = self.args.splicemix_start_epoch
            if epoch == target_epoch or epoch == target_epoch - 1:
                full_purities = torch.cat(all_purities).detach().cpu().numpy()
                plot_knn_purity(full_purities, class_idx=c, epoch=epoch, threshold=current_thresh, save_dir=vis_dir)
                
        return label_clean_matrix
    def gmm_cleaning_center(self, feats, targets,epoch):
        """
        对每个类别单独进行 GMM 聚类，区分 Clean/Noisy 标签
        """
        print("[RoLT] Running GMM Cleaning per class...")
        # 初始化全为 False (Noisy)
        label_clean_matrix = torch.zeros_like(targets, dtype=torch.bool).to(self.device)
        
        for c in range(self.num_classes):
            pos_indices = (targets[:, c] == 1)
            num_samples = pos_indices.sum().item()
            
            # 样本太少无法拟合 GMM，直接视为 Clean
            if num_samples < 100: 
                label_clean_matrix[pos_indices, c] = True
                continue
                
            # 获取特征并归一化
            class_feats = feats[pos_indices, c, :]
            class_feats = F.normalize(class_feats, p=2, dim=1)
            
            # 计算到原型的 Cosine Distance (1 - Cosine Similarity)
            # 或者 Euclidean Distance。RoLT 常用 Cosine。
            # 这里原型和特征都已经归一化，Euclidean^2 = 2 - 2*Cosine
            # 所以直接用 1 - (feat * proto) 作为距离度量
            sim = torch.matmul(class_feats, self.prototypes[c].unsqueeze(-1)).squeeze() # [N_pos]
            dists = 1.0 - sim
            dists = dists.cpu().numpy().reshape(-1, 1)
            
            # GMM 拟合 (2 components)
            try:
                gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4,random_state=self.args.seed)#初始化一个高斯混合模型
                gmm.fit(dists)
                # ================= 可视化调用 =================
                # 动态获取 Stage 1 准备结束时的 Epoch (如果你的 epoch 从 0 开始算，这里可能是 target_epoch - 1)
                # 为了保险起见，我们可以在 epoch 等于 target_epoch 或 target_epoch-1 时都触发
                target_epoch = self.args.splicemix_start_epoch
                
                # 既然是最后阶段的验收，直接去掉 (c == 0 or ...) 的限制，为所有疾病类别画图！
                if epoch == target_epoch or epoch == target_epoch - 1:
                    base_out_dir = self.args.output
                    save_path = os.path.join(base_out_dir, "gmm_visualizations")
                    plot_distance_gmm_fit(dists, gmm, class_idx=c, epoch=epoch,save_dir=save_path)
                # ==============================================
                # 均值较小的 Gaussian 分布对应 Clean (距离原型近)
                clean_comp_idx = gmm.means_.argmin()
                
                # 获取每个样本属于 Clean 分布的概率 post_prob
                probs = gmm.predict_proba(dists)
                prob_clean = probs[:, clean_comp_idx]
                
                # 设定阈值 (如 0.5) 判定是否 Clean，在二分类的高斯混合模型中，0.5 是最自然的决策边界。
                is_clean_subset = (prob_clean > 0.5)
                # 将结果填回大矩阵
                # 需要复杂的索引映射
                full_indices = torch.where(pos_indices)[0]
                # 仅将判定为 Clean 的位置设为 True
                clean_indices = full_indices[is_clean_subset]
                label_clean_matrix[clean_indices, c] = True
                
            except Exception as e:
                # GMM 失败 (如数据分布奇异)，保守策略：全部视为 Clean
                print(f"Warning: GMM failed for class {c}, setting all to clean. Error: {e}")
                label_clean_matrix[pos_indices, c] = True
                
        return label_clean_matrix
    def gmm_cleaning_loss(self, losses, targets):
        """
        基于 Small-Loss Criterion 的 GMM 清洗
        """
        print("[RoLT] Running GMM Cleaning per class (using Small-Loss Criterion)...")
        # 初始化全为 False (Noisy)
        label_clean_matrix = torch.zeros_like(targets, dtype=torch.bool).to(self.device)
        
        for c in range(self.num_classes):
            pos_indices = (targets[:, c] == 1)
            num_samples = pos_indices.sum().item()
            
            # 样本太少无法拟合 GMM，直接视为 Clean
            if num_samples < 100: 
                label_clean_matrix[pos_indices, c] = True
                continue
                
            # 获取该类别正样本的 Loss 值，作为 GMM 的一维输入
            class_losses = losses[pos_indices, c]
            
            # 将 Loss 转换为 GMM 要求的 numpy 格式
            dists = class_losses.detach().cpu().numpy().reshape(-1, 1)
            
            # GMM 拟合 (2 components)
            try:
                gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=self.args.seed)
                gmm.fit(dists)
                
                # 均值较小的 Gaussian 分布对应 Clean (Loss 越小越干净)
                clean_comp_idx = gmm.means_.argmin()
                
                # 获取每个样本属于 Clean 分布的概率 post_prob
                probs = gmm.predict_proba(dists)
                prob_clean = probs[:, clean_comp_idx]
                
                # 设定阈值 0.5 判定是否 Clean
                is_clean_subset = (prob_clean > 0.5)
                
                # 填回大矩阵
                full_indices = torch.where(pos_indices)[0]
                clean_indices = full_indices[is_clean_subset]
                label_clean_matrix[clean_indices, c] = True
                
            except Exception as e:
                # GMM 失败的保守策略
                print(f"Warning: GMM failed for class {c}, setting all to clean. Error: {e}")
                label_clean_matrix[pos_indices, c] = True
        return label_clean_matrix
    def apply_sample_selection_rule(self, label_clean_matrix, targets, indices):
        """
        严格且安全的筛选规则
        1. 保护无病样本 (No Finding)
        2. 有病样本坚持 Clean >= Noisy
        """
        print("[RoLT] Applying STRICT Sample Selection Rule (Protecting 'No Finding')...")
        
        # 仅考虑正标签 (targets=1)
        # label_clean_matrix 中 targets=0 的位置虽然是 False，但不应计入 Noisy Count
        # 所以我们需要一个 valid_mask
        valid_pos_mask = (targets == 1)
        # [修复核心] 找出所有健康的片子 (14个标签全为0的样本)
        no_finding_mask = (targets.sum(dim=1) == 0)
        # 统计 Clean Positive Tags
        # (label_clean_matrix 为 True 且 valid_pos_mask 为 True)
        clean_counts = (label_clean_matrix & valid_pos_mask).sum(dim=1)
        
        # 统计 Noisy Positive Tags
        # (label_clean_matrix 为 False 且 valid_pos_mask 为 True)
        noisy_counts = ((~label_clean_matrix) & valid_pos_mask).sum(dim=1)
        
        # 执行判定
        is_sample_clean = no_finding_mask |((clean_counts) > noisy_counts)
        # 保留条件：要么是健康片子(没有标签)，要么是带病片子且正确的标签 >= 错误的标签
        # is_sample_clean = no_finding_mask | (clean_counts >= noisy_counts)
        
        # 统计结果
        # 2. [新增] 附加判断条件
        # 统计每个样本的总正标签数
        # total_pos_tags = clean_counts + noisy_counts
        #     # 识别出：只有一个标签 且 该标签是噪声的样本
        #     # (此时 clean_counts 为 0, noisy_counts 为 1)
        # is_single_noisy = (total_pos_tags == 1) & (noisy_counts == 1)
        #     # 将这些样本强制设为 False (Noisy)
        #     # 使用位运算: 原结果 AND (NOT is_single_noisy)
        # is_sample_clean = is_sample_clean & (~is_single_noisy)
        
        num_clean = is_sample_clean.sum().item()
        print(f"[RoLT] Trustworthy (Clean) Samples found: {num_clean} / {len(targets)}")
        
        # 构建字典返回
        clean_mask_dict = {}
        idx_cpu = indices.cpu().numpy()
        mask_cpu = is_sample_clean.cpu().numpy()
        for idx, is_clean in zip(idx_cpu, mask_cpu):
            clean_mask_dict[idx] = bool(is_clean)
            
        return clean_mask_dict

    def refine_prototypes(self, feats, targets, indices, clean_mask_dict):
        """
        仅使用被判定为 Trustworthy (Clean Sample) 且标签本身 Clean 的样本
        再次更新原型
        """
        # 构建 Boolean Mask 向量
        is_sample_trustworthy = torch.tensor(
            [clean_mask_dict[idx.item()] for idx in indices]
        ).bool().to(self.device)
        
        # 我们只希望用 "高质量样本" 更新原型
        # 策略：Filter feats based on is_sample_trustworthy
        
        new_prototypes = torch.zeros_like(self.prototypes)
        
        for c in range(self.num_classes):
            # 条件1: 样本包含该标签 (targets=1)
            # 条件2: 样本整体是 Trustworthy 的
            valid_indices = (targets[:, c] == 1) & is_sample_trustworthy
            
            if valid_indices.sum() > 0:
                class_feats = feats[valid_indices, c, :]
                class_feats = F.normalize(class_feats, p=2, dim=1)
                proto = class_feats.mean(dim=0)
                new_prototypes[c] = F.normalize(proto, p=2, dim=0)
            else:
                # 如果没有 Trustworthy 样本，保持上一轮原型
                new_prototypes[c] = self.prototypes[c]
                
        self.prototypes = new_prototypes

    def generate_soft_labels(self, feats, logits, targets, indices, clean_mask_dict, label_clean_matrix):
        """
        生成软伪标签 (Soft Pseudo Labels) - 自定义混合比例
        结合细粒度 GMM 清洗矩阵与假阴性(漏标)恢复策略
        """
        print("[RoLT] Generating Soft Pseudo Labels (Custom Mix: 0.1*ERM + 0.8*NCM + 0.1*Uni with Fine-Grained Replacement)...")
        
        soft_label_dict = {}
            
        # 1. 找出 Noisy Samples (即整体被判定为包含噪声或需要检查的样本)
        is_sample_trustworthy = torch.tensor(
            [clean_mask_dict[idx.item()] for idx in indices]
        ).bool().to(self.device)
        
        noisy_indices_loc = torch.where(~is_sample_trustworthy)[0]
        
        if len(noisy_indices_loc) == 0:
            return soft_label_dict
                
        # 提取 Noisy 样本的相关数据
        noisy_feats = feats[noisy_indices_loc]     # [M, C, D]
        noisy_logits = logits[noisy_indices_loc]   # [M, C]
        noisy_targets = targets[noisy_indices_loc] # [M, C] (原始标签)
        
        # # --- 组件 1: ERM 预测 (0.1 权重) ---
        # p_erm = torch.sigmoid(noisy_logits)
            
        # # --- 组件 2: NCM 预测 (0.8 权重) ---
        # norm_feats = F.normalize(noisy_feats, p=2, dim=2)
        # # sim[i, c] = dot(feat[i,c], proto[c])
        # sims = torch.einsum('ncd,cd->nc', norm_feats, self.prototypes)
        
        # # 【核心修复】引入 margin，将决策边界右移，防止正交(不相关)特征获得过高的基线概率
        # margin = 0.75 
        # tau = 10.0 
        # p_ncm = torch.sigmoid((sims - margin) * tau)
        
        # # --- 组件 3: 均匀分布 (0.1 权重) ---
        # # 医疗数据集正样本极少，使用 0.05 作为安全基线
        # p_uniform = 0.05
        
        # # --- 纯计算出的软标签张量 ---
        # pure_soft_targets = (0.1 * p_erm) + (0.8 * p_ncm) + (0.1 * p_uniform)
        
        # # ================= 细粒度标签替换逻辑 =================
        
        # # 1. 获取这些 Noisy 样本在 GMM 中的细粒度 Clean/Noisy 矩阵
        # fine_grained_clean_mask = label_clean_matrix[noisy_indices_loc] 
        
        # # 2. 条件1：原本是阳性（1），但被判定为噪声（假阳性）
        # is_noisy_positive = (noisy_targets == 1) & (~fine_grained_clean_mask)
        
        # # 3. 条件2：原本是阴性（0），但软标签算出来大于 0.85（疑似假阴性漏标）
        # # 如果模型极其确信这里有病灶，我们将其纳入替换范围
        # is_suspected_false_negative = (noisy_targets == 0) & (pure_soft_targets > 0.85)& (p_erm > 0.3)
        
        # # 4. 合并替换条件：只要满足上面任意一种，就属于需要“被纠正”的标签
        # should_replace = is_noisy_positive | is_suspected_false_negative
        
        # # 5. 执行细粒度替换：
        # # 满足条件的用 pure_soft_targets 覆盖；不满足的保留原始 target (绝对的 0 和 1)
        # final_soft_targets = torch.where(should_replace, pure_soft_targets, noisy_targets.float())
        # # =======================================================

        p_erm = torch.sigmoid(noisy_logits)
            
        norm_feats = F.normalize(noisy_feats, p=2, dim=2)
        sims = torch.einsum('ncd,cd->nc', norm_feats, self.prototypes)
        
        # 保持 0.5 的阈值，这是余弦相似度极佳的分水岭
        margin = 0.5 
        tau = 10.0 
        p_ncm = torch.sigmoid((sims - margin) * tau)
        
        # 【保留】温和缓冲：防止 BCE Loss 梯度爆炸
        pure_soft_targets = (0.5 * p_erm) + (0.5 * p_ncm)
        
        # KNN 已经极其精准地给出了判定结果
        fine_grained_clean_mask = label_clean_matrix[noisy_indices_loc] 
        
        # 【修改】卸下免死金牌！全权信任 KNN 的局部纯度判定！
        # 只要原始是 1，且 KNN 说是噪声（False），就大胆替换！
        is_noisy_positive = (noisy_targets == 1) & (~fine_grained_clean_mask)
        
        # 【保留】假阴性漏标恢复的严苛双重锁
        is_suspected_false_negative = (noisy_targets == 0) & (pure_soft_targets > 0.85) & (p_erm > 0.3)
        
        # 合并与执行替换
        should_replace = is_noisy_positive | is_suspected_false_negative
        final_soft_targets = torch.where(should_replace, pure_soft_targets, noisy_targets.float())
        # 存入字典返回
        noisy_real_indices = indices[noisy_indices_loc]
        
        for idx, s_label in zip(noisy_real_indices, final_soft_targets):
            # 存入 CPU 字典以节省显存
            soft_label_dict[idx.item()] = s_label.detach().cpu()
      
        return soft_label_dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import shutil
import os

# ==========================================
# 1. 构造 Mock 模型 (模拟 Q2L 模型的输出)
# ==========================================
class MockQ2LModel(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.dummy_param = nn.Parameter(torch.randn(1)) 

    def forward(self, x):
        batch_size = x.size(0)
        # 【修复点】：增加 device=x.device，确保生成的张量和输入图片在同一个设备(GPU或CPU)
        logits = torch.randn(batch_size, self.num_classes, device=x.device)
        features = torch.randn(batch_size, self.num_classes, self.feature_dim, device=x.device)
        return logits, features, None, None

# ==========================================
# 2. 构造 Mock 数据集 (模拟医疗图像数据)
# ==========================================
class MockDataset(Dataset):
    def __init__(self, num_samples, num_classes):
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 模拟图像输入 [C, H, W]
        image = torch.randn(3, 224, 224)
        # 模拟多标签 Target [Num_Classes], 随机生成 0 或 1
        # 为了测试保护 "No Finding" 逻辑，我们故意让一部分全为 0
        if idx % 10 == 0:
            target = torch.zeros(self.num_classes, dtype=torch.long)
        else:
            target = torch.randint(0, 2, (self.num_classes,), dtype=torch.long)
        return image, target, idx

# ==========================================
# 3. 主调试流程
# ==========================================
def run_debug():
    print(">>> 开始准备调试环境...")
    
    # 清理历史作图文件夹，防止干扰
    plot_dir = "./gmm_distance_plots"
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
        
    # 设置超参数
    NUM_CLASSES = 5       # 减少类别以加快测试速度
    FEATURE_DIM = 256     # 缩小维度
    NUM_SAMPLES = 400     # 必须大于 100 以触发 GMM 逻辑
    BATCH_SIZE = 32
    
    # 模拟 args
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.seed = 42
    args.splicemix_start_epoch = 2  # 设置目标 epoch 为 2，触发画图逻辑
    
    # 实例化数据与模型
    dataset = MockDataset(num_samples=NUM_SAMPLES, num_classes=NUM_CLASSES)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = MockQ2LModel(num_classes=NUM_CLASSES, feature_dim=FEATURE_DIM)
    
    # 将模型放入 GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 实例化你的 RoLT_Handler
    handler = RoLT_Handler(
        args=args,
        model=model,
        train_loader=train_loader,
        num_classes=NUM_CLASSES,
        feature_dim=FEATURE_DIM
    )
    
    # ---------------------------------------------------------
    # 测试 1: 测试基于特征的 step 方法 (Epoch 1 将触发画图，因为 epoch == target_epoch - 1)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(">>> 测试 1: handler.step(epoch=1) [基于特征的 GMM]")
    print("="*50)
    clean_mask_dict, soft_label_dict = handler.step(epoch=1)
    
    print(f"-> 成功返回 clean_mask_dict，包含 {len(clean_mask_dict)} 个样本的状态。")
    print(f"-> 成功返回 soft_label_dict，包含 {len(soft_label_dict)} 个需要纠正的软标签。")
    if os.path.exists(plot_dir):
        print(f"-> GMM 画图已触发！图片保存在: {plot_dir}")
    
    # ---------------------------------------------------------
    # 测试 2: 测试基于 Loss 的 step_loss 方法 (Epoch 2 再次触发画图)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(">>> 测试 2: handler.step_loss(epoch=2) [基于 Small-Loss 的 GMM]")
    print("="*50)
    clean_mask_dict_loss, soft_label_dict_loss = handler.step_loss(epoch=2)
    
    print(f"-> 成功返回 clean_mask_dict_loss，包含 {len(clean_mask_dict_loss)} 个样本的状态。")
    print(f"-> 成功返回 soft_label_dict_loss，包含 {len(soft_label_dict_loss)} 个需要纠正的软标签。")
    
    print("\n>>> 所有的核心逻辑均已走通，没有发生崩溃！")

if __name__ == '__main__':
    run_debug()