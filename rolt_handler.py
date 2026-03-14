import torch
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import math

class RoLT_Handler:
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
        label_clean_matrix = self.gmm_cleaning(all_feats, all_targets)
        
        # 4. 样本级筛选 (+2 规则)
        # 判定哪些样本有资格进入 "SpliceMix" (即 Clean 样本)
        # 返回: {sample_index: is_clean_sample(bool)}
        clean_mask_dict = self.apply_sample_selection_rule(label_clean_matrix, all_targets, all_indices)
        
        # 5. 基于筛选出的“值得信赖”样本，二次更新原型 (Refinement)
        self.refine_prototypes(all_feats, all_targets, all_indices, clean_mask_dict)
        
        # 6. 生成软伪标签 (Soft Pseudo Labels)
        # 为“噪声样本”生成用于校正的软标签
        # 返回: {sample_index: soft_targets(Tensor)}
        soft_label_dict = self.generate_soft_labels(all_feats, all_logits, all_targets, all_indices, clean_mask_dict)        
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
                mask = targets.unsqueeze(-1)
                masked_features = features * mask 
                
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

    def gmm_cleaning(self, feats, targets):
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
        is_sample_clean = no_finding_mask |((clean_counts+1) > noisy_counts)
        # 保留条件：要么是健康片子(没有标签)，要么是带病片子且正确的标签 >= 错误的标签
        # is_sample_clean = no_finding_mask | (clean_counts >= noisy_counts)
        
        # 统计结果
        # 2. [新增] 附加判断条件
        # 统计每个样本的总正标签数
        # total_pos_tags = clean_counts + noisy_counts
        #     # 识别出：只有一个标签 且 该标签是噪声 的样本
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

    def generate_soft_labels(self, feats, logits, targets, indices, clean_mask_dict):
        """
        生成软伪标签 (Soft Pseudo Labels) - 自定义混合比例
        Formula: 0.4 * ERM + 0.2 * NCM + 0.2 * Original + 0.2 * Uniform
        """
        print("[RoLT] Generating Soft Pseudo Labels (Custom Mix: 0.4*ERM + 0.2*NCM + 0.2*Orig + 0.2*Uni)...")
        
        soft_label_dict = {}
        
        # 1. 找出 Noisy Samples (即被判定为不 Clean 的样本)
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
        
        # --- 组件 1: ERM 预测 (0.4) ---
        p_erm = torch.sigmoid(noisy_logits)
        
        # --- 组件 2: NCM 预测 (0.2) ---
        norm_feats = F.normalize(noisy_feats, p=2, dim=2)
        # sim[i, c] = dot(feat[i,c], proto[c])
        sims = torch.einsum('ncd,cd->nc', norm_feats, self.prototypes)
        tau = 10.0 
        p_ncm = torch.sigmoid(sims * tau)
        
        # --- 组件 3: 原始标签 (0.2) ---
        p_orig = noisy_targets.float()
        
        # --- 组件 4: 均匀分布 (0.2) ---
        # 对于多标签二分类(BCE)，均匀分布/最大不确定性意味着概率为 0.5
        # 0.5 太高了！在 NIH 数据集中，正样本比例极低，0.05 是更合理的基线
        p_uniform = 0.05
        
        # --- 执行加权混合 ---
        # 比例: 0.4 : 0.2 : 0.2 : 0.2
        # soft_targets = (0.4 * p_erm) + (0.2 * p_ncm) + (0.2 * p_orig) + (0.2 * p_uniform)
        # [核心修改] 新的权重分配
        # 0.7 信任原型特征，0.2 信任模型当前预测，0.1 进行微量平滑
        soft_targets = (0.1 * p_erm) + (0.8 * p_ncm) + (0.1 * p_uniform)
        # 存入字典
        noisy_real_indices = indices[noisy_indices_loc]
        
        for idx, s_label in zip(noisy_real_indices, soft_targets):
            # 存入 CPU 字典以节省显存
            soft_label_dict[idx.item()] = s_label.detach().cpu()
            
        return soft_label_dict