import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset

# - 增加过采样：对少数类进行过采样，使每种故障类型的样本数达到无故障样本的10%，同时至少是原来的3倍
# - 调整类别权重：增加无故障类别的权重，使模型更关注无故障样本
# - 修改模型结构：增强特征嵌入层，使模型能更好地学习特征表示
# - 调整超参数：降低学习率，增加权重衰减，调整焦点损失参数
# - 添加训练进度跟踪：更好地监控训练过程
# - 移除TWF和RNF专用分类器：简化模型，只保留多分类和二分类功能
# - 平衡准确率和召回率：调整阈值和损失函数，提高故障类型的精确率

# 导入Mamba模型
from model import ModelArgs, Mamba, ResidualBlock, MambaBlock, RMSNorm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mamba_training.log'),
        logging.StreamHandler()
    ]
)

# 设置 Matplotlib 的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义焦点损失函数，更好地处理类别不平衡问题
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# 定义带有精确率惩罚的损失函数
class PrecisionFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean', precision_penalty=0.3):
        super(PrecisionFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.precision_penalty = precision_penalty
        self.base_loss = FocalLoss(alpha, gamma, weight, 'none')
        
    def forward(self, inputs, targets):
        # 基础焦点损失
        focal_loss = self.base_loss(inputs, targets)
        
        # 计算每个样本的预测类别
        _, preds = torch.max(inputs, 1)
        
        # 对于预测为故障但实际为无故障的样本增加惩罚
        # 这有助于提高精确率
        is_fault_pred = (preds > 0)
        is_normal_true = (targets == 0)
        false_positives = is_fault_pred & is_normal_true
        
        # 增加惩罚
        focal_loss[false_positives] = focal_loss[false_positives] * (1 + self.precision_penalty)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


@dataclass
class MambaFaultConfig:
    """配置类"""
    data_path: str = "./ai4i2020.csv"
    model_save_path: str = "best_mamba_model.pth"
    base_output_dir: str = "mamba_experiment_results"  # 基础输出目录
    experiment_data_path: str = "mamba_evaluation_results.json"
    batch_size: int = 32  # 批量大小
    num_epochs: int = 300  # 训练轮次
    early_stopping_patience: int = 50  # 早停耐心值

    # 模型参数
    d_model: int = 128  # 模型维度
    n_layer: int = 4  # 层数
    d_state: int = 16  # 状态空间维度
    expand: int = 2  # 扩展因子
    dropout: float = 0.2  # dropout率

    # 修改学习率和权重衰减
    learning_rate: float = 3e-4  # 降低学习率
    weight_decay: float = 2e-3  # 增加权重衰减
    
    # 类别权重
    use_class_weights: bool = True
    
    # 多任务学习
    use_multitask: bool = True
    # 多任务损失权重
    multi_task_weight: float = 0.7  # 多分类任务权重
    binary_task_weight: float = 0.3  # 二分类任务权重
    
    # 焦点损失参数
    use_focal_loss: bool = True  # 使用焦点损失
    focal_gamma: float = 3.0  # 增加焦点损失gamma参数，更关注难分类样本
    focal_alpha: float = 0.3  # 调整焦点损失alpha参数
    
    # 精确率惩罚参数
    use_precision_penalty: bool = True  # 使用精确率惩罚
    precision_penalty: float = 0.5  # 精确率惩罚系数
    
    # 特征工程
    use_advanced_features: bool = True  # 使用高级特征工程
    
    # 数据增强
    use_mixup: bool = True  # 使用Mixup数据增强
    mixup_alpha: float = 0.2  # Mixup参数
    
    # 增加过采样参数
    use_oversampling: bool = True  # 使用过采样
    oversampling_ratio: float = 0.15  # 过采样比例（相对于无故障样本）
    min_oversampling_factor: int = 4  # 最小过采样倍数
    
    # 预测阈值调整
    prediction_threshold: float = 0.6  # 预测为故障的概率阈值
    
    device: torch.device = None
    columns_to_drop: List[str] = None
    output_dir: str = None  # 将在 __post_init__ 中设置
    fault_types: List[str] = None  # 故障类型列表
    num_classes: int = 6  # 包括无故障和5种故障类型(HDF, PWF, OSF, TWF, RNF)

    def __post_init__(self):
        self.columns_to_drop = ['UDI', 'Product ID', 'TWF', 'RNF']  # 只删除不相关的ID列
        self.fault_types = ['HDF', 'PWF', 'OSF']  # 考虑所有5种故障类型

        # 创建基础输出目录
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)

        # 使用当前时间创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.base_output_dir, timestamp)

        # 创建实验目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 更新模型保存路径
        self.model_save_path = os.path.join(self.output_dir, self.model_save_path)

        # 检测并使用最快的可用设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # 设置 CUDA 设备的随机种子
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)  # 如果使用多GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logging.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logging.info("使用 CPU")


class MambaFaultPredictionModel(nn.Module):
    """基于Mamba的故障预测模型"""
    def __init__(self, config: MambaFaultConfig, input_dim: int, num_classes: int):
        super().__init__()
        self.config = config
        
        # 创建Mamba模型参数
        mamba_args = ModelArgs(
            d_model=config.d_model,
            n_layer=config.n_layer,
            vocab_size=num_classes,
            d_state=config.d_state,
            expand=config.expand
        )
        
        # 特征嵌入层 - 增强特征提取能力
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, config.d_model * 2),
            nn.LayerNorm(config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model * 2),  # 增加一层
            nn.LayerNorm(config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
    
        # Mamba层
        self.layers = nn.ModuleList([ResidualBlock(mamba_args) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model)
        
        # 分类头 - 增强分类能力
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, num_classes)
        )
        
        # 二分类头（用于多任务学习）
        if config.use_multitask:
            self.binary_classifier = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.LayerNorm(config.d_model // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, 2)  # 二分类：有故障/无故障
            )
    
    def forward(self, x):
        # x: [batch_size, features]
        
        # 将输入转换为序列形式 [batch_size, 1, features]
        x = x.unsqueeze(1)
        
        # 特征嵌入 [batch_size, 1, d_model]
        x = self.embedding(x)
        
        # 通过Mamba层
        for layer in self.layers:
            x = layer(x)
        
        # 最终归一化
        x = self.norm_f(x)
        
        # 取序列的第一个元素作为分类输入
        x = x.squeeze(1)
        
        # 主分类器
        logits = self.classifier(x)
        
        # 如果使用多任务学习，返回多分类和二分类的结果
        if self.config.use_multitask:
            binary_logits = self.binary_classifier(x)
            return logits, binary_logits
        else:
            return logits


class MambaDataProcessor:
    """数据处理类"""

    def __init__(self, config: MambaFaultConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()  # 用于编码产品类型
        self.fault_mapping = None  # 故障类型映射
        self.class_weights = None  # 类别权重

    def load_and_preprocess(self) -> Tuple[DataLoader, DataLoader, torch.Tensor, np.ndarray, Dict[str, Any]]:
        # 加载数据
        try:
            df = pd.read_csv(self.config.data_path)
            logging.info("数据加载成功！")
        except FileNotFoundError:
            logging.error(f"错误：文件未找到，请检查路径 {self.config.data_path}")
            raise

        # 数据探索
        logging.info(f"数据集形状: {df.shape}")
        logging.info(f"数据集列: {df.columns.tolist()}")
        
        # 特征工程 - 增加更多工程特征
        if self.config.use_advanced_features:
            # 添加温差特征
            df['Temp_diff'] = df['Air temperature [K]'] - df['Process temperature [K]']

            # 添加旋转力矩与转速的比率
            df['Torque_Rotational_ratio'] = df['Torque [Nm]'] / (df['Rotational speed [rpm]'] + 1e-8)
            
            # 添加功率特征 (转速 * 扭矩的近似值)
            df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']
            
            # 添加工具磨损率 (工具磨损 / 过程温度)
            df['Tool_wear_rate'] = df['Tool wear [min]'] / (df['Process temperature [K]'] + 1e-8)
            
            # 添加温度与工具磨损的交互特征
            df['Temp_Tool_interaction'] = df['Process temperature [K]'] * df['Tool wear [min]']
            
            # 添加转速与工具磨损的交互特征
            df['Rotation_Tool_interaction'] = df['Rotational speed [rpm]'] * df['Tool wear [min]']
            
            # 添加力矩与工具磨损的交互特征
            df['Torque_Tool_interaction'] = df['Torque [Nm]'] * df['Tool wear [min]']
            
            # 添加非线性特征
            df['Rotational_speed_squared'] = df['Rotational speed [rpm]'] ** 2
            df['Tool_wear_squared'] = df['Tool wear [min]'] ** 2
            df['Torque_squared'] = df['Torque [Nm]'] ** 2
            
            # 添加新的特征组合，有助于区分故障类型
            df['Process_temp_tool_wear_ratio'] = df['Process temperature [K]'] / (df['Tool wear [min]'] + 1e-8)
            df['Air_temp_tool_wear_ratio'] = df['Air temperature [K]'] / (df['Tool wear [min]'] + 1e-8)
            df['Rotational_torque_temp_interaction'] = df['Rotational speed [rpm]'] * df['Torque [Nm]'] / (df['Process temperature [K]'] + 1e-8)
            
            logging.info("添加高级工程特征完成")
        
        # 编码产品类型
        df['Type_encoded'] = self.label_encoder.fit_transform(df['Type'])
        logging.info(f"产品类型编码映射: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # 创建故障类型标签
        # 首先检查每行是否有故障
        df['Has_Failure'] = df['Machine failure'].astype(bool)
        
        # 创建故障类型标签（0表示无故障，1-5表示不同类型的故障）
        df['Fault_Type'] = 0  # 默认为无故障
        
        # 为每种故障类型分配一个唯一的标签
        for i, fault_type in enumerate(self.config.fault_types, 1):
            mask = df[fault_type] == 1
            df.loc[mask, 'Fault_Type'] = i
        
        # 创建故障类型映射
        self.fault_mapping = {0: '无故障'}
        for i, fault_type in enumerate(self.config.fault_types, 1):
            self.fault_mapping[i] = fault_type
        
        logging.info(f"故障类型映射: {self.fault_mapping}")
        logging.info(f"故障类型分布:\n{df['Fault_Type'].value_counts().sort_index()}")
        
        # 数据预处理
        # 删除不需要的列，保留Type_encoded作为特征，但删除原始Type列
        # 同时删除Machine failure和故障类型列，因为它们是预测目标
        columns_to_drop = self.config.columns_to_drop + ['Type', 'Machine failure'] + self.config.fault_types
        
        # 提取特征和标签
        X = df.drop(columns=columns_to_drop + ['Fault_Type', 'Has_Failure']).copy()
        y = df['Fault_Type'].values
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 计算类别权重 - 修改权重计算方式，更加平衡精确率和召回率
        class_counts = np.bincount(y)
        total_samples = len(y)
        
        # 使用更合理的权重计算方式，增加故障类别的权重以提高精确率
        class_weights = total_samples / (len(class_counts) * class_counts)
        
        # 对无故障类别进行特殊处理，增加其权重
        # 无故障类别索引为0
        class_weights[0] = class_weights[0] * 2.0  # 降低无故障类别的权重，提高故障类别的精确率
        
        # 对故障类别进行特殊处理，增加其权重以提高精确率
        for i in range(1, len(class_weights)):
            class_weights[i] = class_weights[i] * 3.0  # 增加故障类别的权重
        
        self.class_weights = torch.FloatTensor(class_weights).to(self.config.device)
        logging.info(f"类别权重: {class_weights}")
        
        # 数据集划分
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 对训练数据进行过采样
        if self.config.use_oversampling:
            X_train_resampled = []
            y_train_resampled = []
            
            # 获取无故障样本数量
            normal_samples = sum(y_train == 0)
            
            # 保留所有无故障样本
            X_train_normal = X_train[y_train == 0]
            y_train_normal = y_train[y_train == 0]
            X_train_resampled.append(X_train_normal)
            y_train_resampled.append(y_train_normal)
            
            # 对每种故障类型进行过采样
            for fault_idx in range(1, len(self.fault_mapping)):
                X_fault = X_train[y_train == fault_idx]
                y_fault = y_train[y_train == fault_idx]
                
                # 计算过采样数量，使每种故障类型的样本数为无故障样本的指定比例
                n_samples = max(int(normal_samples * self.config.oversampling_ratio), 
                               len(X_fault) * self.config.min_oversampling_factor)
                
                # 过采样
                if len(X_fault) > 0:  # 确保有样本可以过采样
                    X_fault_resampled, y_fault_resampled = resample(
                        X_fault, y_fault, 
                        replace=True, 
                        n_samples=n_samples, 
                        random_state=42
                    )
                    X_train_resampled.append(X_fault_resampled)
                    y_train_resampled.append(y_fault_resampled)
            
            # 合并所有样本
            X_train = np.vstack(X_train_resampled)
            y_train = np.concatenate(y_train_resampled)
            
            logging.info(f"过采样后的训练集分布:\n{pd.Series(y_train).value_counts().sort_index()}")
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.config.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.config.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.config.device)
        
        # 创建二分类标签（有故障/无故障）
        y_train_binary = (y_train > 0).astype(int)
        y_train_binary_tensor = torch.LongTensor(y_train_binary).to(self.config.device)
        
        # 创建数据加载器 - 只包含多分类和二分类标签
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, y_train_binary_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # 返回数据加载器和测试数据
        feature_info = {
            'input_dim': X_train.shape[1],
            'num_classes': len(self.fault_mapping)
        }
        
        return train_loader, None, X_test_tensor, y_test, feature_info


# Mixup数据增强
def mixup_data(x, y, alpha=1.0, device=None):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失计算"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class MambaModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: MambaFaultConfig, model: MambaFaultPredictionModel, class_weights=None):
        self.config = config
        self.model = model
        self.class_weights = class_weights
        
        # 设置优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 设置学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=False
        )
        
        # 设置损失函数
        if config.use_focal_loss:
            if config.use_precision_penalty:
                self.criterion_multi = PrecisionFocalLoss(
                    alpha=config.focal_alpha,
                    gamma=config.focal_gamma,
                    weight=self.class_weights if config.use_class_weights else None,
                    precision_penalty=config.precision_penalty
                )
            else:
                self.criterion_multi = FocalLoss(
                    alpha=config.focal_alpha,
                    gamma=config.focal_gamma,
                    weight=self.class_weights if config.use_class_weights else None
                )
        else:
            self.criterion_multi = nn.CrossEntropyLoss(
                weight=self.class_weights if config.use_class_weights else None
            )
        
        self.criterion_binary = nn.CrossEntropyLoss()
        
        # 早停设置
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logging.info("模型训练器初始化完成")   
 
    def train(self, train_loader, X_val, y_val):
        """训练模型"""
        train_losses = []
        val_losses = []
        
        # 将验证数据转换为张量
        y_val_tensor = torch.LongTensor(y_val).to(self.config.device)
        
        # 创建二分类验证标签
        y_val_binary = (y_val > 0).astype(int)
        y_val_binary_tensor = torch.LongTensor(y_val_binary).to(self.config.device)
        
        logging.info(f"开始训练，总轮次: {self.config.num_epochs}")
        
        for epoch in range(self.config.num_epochs):
            # 训练阶段
            self.model.train()
            epoch_loss = 0.0
            
            # 添加进度跟踪
            batch_count = len(train_loader)
            
            for batch_idx, (data, target, target_binary) in enumerate(train_loader):
                # 每10个批次打印一次进度
                if batch_idx % 10 == 0:
                    logging.info(f"轮次 {epoch+1}/{self.config.num_epochs}, 批次 {batch_idx}/{batch_count}")
                
                self.optimizer.zero_grad()
                
                # 应用Mixup数据增强
                if self.config.use_mixup and np.random.random() < 0.5:  # 50%的概率使用Mixup
                    data, target_a, target_b, lam = mixup_data(
                        data, target, self.config.mixup_alpha, self.config.device
                    )
                    
                    # 前向传播
                    outputs = self.model(data)
                    
                    # 多任务学习
                    if self.config.use_multitask:
                        logits, binary_logits = outputs
                        
                        # 计算多分类损失（使用Mixup）
                        loss_multi = mixup_criterion(
                            self.criterion_multi, logits, target_a, target_b, lam
                        )
                        
                        # 计算二分类损失（不使用Mixup）
                        loss_binary = self.criterion_binary(binary_logits, target_binary)
                        
                        # 总损失
                        loss = (self.config.multi_task_weight * loss_multi + 
                                self.config.binary_task_weight * loss_binary)
                    else:
                        logits = outputs
                        
                        # 计算多分类损失（使用Mixup）
                        loss = mixup_criterion(
                            self.criterion_multi, logits, target_a, target_b, lam
                        )
                else:
                    # 前向传播
                    outputs = self.model(data)
                    
                    # 多任务学习
                    if self.config.use_multitask:
                        logits, binary_logits = outputs
                        
                        # 计算多分类损失
                        loss_multi = self.criterion_multi(logits, target)                       
                        
                        # 计算二分类损失
                        loss_binary = self.criterion_binary(binary_logits, target_binary)
                        
                        # 总损失
                        loss = (self.config.multi_task_weight * loss_multi + 
                                self.config.binary_task_weight * loss_binary)
                    else:
                        logits = outputs
                        
                        # 计算多分类损失
                        loss = self.criterion_multi(logits, target)
                
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # 计算平均训练损失
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            val_loss, val_metrics = self.evaluate(X_val, y_val, y_val_tensor, y_val_binary_tensor)
            val_losses.append(val_loss)
            
            # 更新学习率调度器
            self.scheduler.step(val_loss)
            
            # 打印训练和验证结果
            logging.info(f"轮次 {epoch+1}/{self.config.num_epochs}, "
                         f"训练损失: {avg_train_loss:.4f}, "
                         f"验证损失: {val_loss:.4f}, "
                         f"验证准确率: {val_metrics['accuracy']:.4f}, "
                         f"验证F1: {val_metrics['f1']:.4f}, "
                         f"验证精确率: {val_metrics['precision']:.4f}, "
                         f"验证召回率: {val_metrics['recall']:.4f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config.model_save_path)
                logging.info(f"保存最佳模型，验证损失: {val_loss:.4f}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            # 早停
            if self.patience_counter >= self.config.early_stopping_patience:
                logging.info(f"早停触发，轮次 {epoch+1}")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(self.config.model_save_path))
        
        return train_losses, val_losses
    
    def evaluate(self, X, y, y_tensor=None, y_binary_tensor=None):
        """评估模型"""
        self.model.eval()
        
        if y_tensor is None:
            y_tensor = torch.LongTensor(y).to(self.config.device)
        
        if y_binary_tensor is None and self.config.use_multitask:
            y_binary = (y > 0).astype(int)
            y_binary_tensor = torch.LongTensor(y_binary).to(self.config.device)
        
        with torch.no_grad():
            # 前向传播
            outputs = self.model(X)
            
            # 多任务学习
            if self.config.use_multitask:
                logits, binary_logits = outputs
                
                # 计算多分类损失
                loss_multi = self.criterion_multi(logits, y_tensor)
                
                # 计算二分类损失
                loss_binary = self.criterion_binary(binary_logits, y_binary_tensor)
                
                # 总损失
                loss = (self.config.multi_task_weight * loss_multi + 
                        self.config.binary_task_weight * loss_binary)
            else:
                logits = outputs
                
                # 计算多分类损失
                loss = self.criterion_multi(logits, y_tensor)
            
            # 获取预测结果
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            
            # 转换为NumPy数组
            preds_np = preds.cpu().numpy()
            probs_np = probs.cpu().numpy()
            
            # 计算评估指标
            accuracy = accuracy_score(y, preds_np)
            precision = precision_score(y, preds_np, average='weighted', zero_division=0)
            recall = recall_score(y, preds_np, average='weighted', zero_division=0)
            f1 = f1_score(y, preds_np, average='weighted', zero_division=0)
            
            # 返回损失和指标
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': preds_np,
                'probabilities': probs_np
            }
            
            return loss.item(), metrics

    def predict(self, X):
        """预测"""
        self.model.eval()
        
        with torch.no_grad():
            # 前向传播
            outputs = self.model(X)
            
            # 多任务学习
            if self.config.use_multitask:
                logits, binary_logits = outputs
                
                # 获取多分类预测结果
                probs = F.softmax(logits, dim=1)
                
                # 获取二分类预测结果
                binary_probs = F.softmax(binary_logits, dim=1)
                
                # 使用阈值调整预测结果
                # 如果二分类预测为无故障（概率大于阈值），则多分类预测也为无故障
                # 这有助于减少假阳性（误报）
                is_normal_binary = binary_probs[:, 0] > self.config.prediction_threshold
                
                # 创建调整后的预测结果
                adjusted_preds = torch.argmax(probs, dim=1)
                adjusted_preds[is_normal_binary] = 0  # 将二分类预测为无故障的样本设为无故障
                
                return adjusted_preds.cpu().numpy(), probs.cpu().numpy(), binary_probs.cpu().numpy()
            else:
                # 获取多分类预测结果
                logits = outputs  # 修复：将outputs赋值给logits变量
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                return preds.cpu().numpy(), probs.cpu().numpy(), None
    
   

class MambaEvaluator:
    """模型评估器"""
    
    def __init__(self, config: MambaFaultConfig, fault_mapping: dict):
        self.config = config
        self.fault_mapping = fault_mapping
    
    def evaluate_and_visualize(self, y_true, y_pred, y_prob, output_dir=None):
        """评估模型并可视化结果"""
        if output_dir is None:
            output_dir = self.config.output_dir
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 计算评估指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 计算每个类别的指标
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # 创建混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 计算归一化混淆矩阵
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 打印评估结果
        logging.info(f"准确率: {accuracy:.4f}")
        logging.info(f"精确率: {precision:.4f}")
        logging.info(f"召回率: {recall:.4f}")
        logging.info(f"F1分数: {f1:.4f}")
        logging.info("\n分类报告:")
        logging.info(classification_report(y_true, y_pred, zero_division=0))
        
        # 可视化混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[self.fault_mapping[i] for i in range(len(self.fault_mapping))],
                   yticklabels=[self.fault_mapping[i] for i in range(len(self.fault_mapping))])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # 可视化归一化混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=[self.fault_mapping[i] for i in range(len(self.fault_mapping))],
                   yticklabels=[self.fault_mapping[i] for i in range(len(self.fault_mapping))])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('归一化混淆矩阵')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'normalized_confusion_matrix.png'))
        plt.close()
        
        # 计算并可视化ROC曲线（一对多）
        plt.figure(figsize=(10, 8))
        
        # 为每个类别计算ROC曲线
        for i in range(len(self.fault_mapping)):
            # 将当前类别视为正类，其他类别视为负类
            y_true_binary = (y_true == i).astype(int)
            y_score = y_prob[:, i]
            
            # 计算ROC曲线
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            # 绘制ROC曲线
            plt.plot(fpr, tpr, lw=2,
                    label=f'{self.fault_mapping[i]} (AUC = {roc_auc:.2f})')
        
        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('ROC曲线')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
        
        # 计算并可视化精确率-召回率曲线
        plt.figure(figsize=(10, 8))
        
        # 为每个类别计算精确率-召回率曲线
        for i in range(len(self.fault_mapping)):
            # 将当前类别视为正类，其他类别视为负类
            y_true_binary = (y_true == i).astype(int)
            y_score = y_prob[:, i]
            
            # 计算精确率-召回率曲线
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, y_score)
            ap = average_precision_score(y_true_binary, y_score)
            
            # 绘制精确率-召回率曲线
            plt.plot(recall_curve, precision_curve, lw=2,
                    label=f'{self.fault_mapping[i]} (AP = {ap:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
        plt.close()
        
        # 保存评估结果
        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': class_report
        }
        
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        
        return evaluation_results


def main():
    """主函数"""
    # 创建配置
    config = MambaFaultConfig()
    
    # 创建数据处理器
    data_processor = MambaDataProcessor(config)
    
    # 加载和预处理数据
    train_loader, _, X_test, y_test, feature_info = data_processor.load_and_preprocess()
    
    # 创建模型
    model = MambaFaultPredictionModel(
        config=config,
        input_dim=feature_info['input_dim'],
        num_classes=feature_info['num_classes']
    ).to(config.device)
    
    # 打印模型结构
    logging.info(f"模型结构:\n{model}")
    
    # 创建训练器
    trainer = MambaModelTrainer(
        config=config,
        model=model,
        class_weights=data_processor.class_weights
    )
    
    # 训练模型
    train_losses, val_losses = trainer.train(train_loader, X_test, y_test)
    
    # 评估模型
    _, metrics = trainer.evaluate(X_test, y_test)
    
    # 获取预测结果
    y_pred, y_prob, _ = trainer.predict(X_test)
    
    # 创建评估器
    evaluator = MambaEvaluator(config, data_processor.fault_mapping)
    
    # 评估和可视化结果
    evaluation_results = evaluator.evaluate_and_visualize(y_test, y_pred, y_prob)

    
    # 保存实验结果到JSON文件
    experiment_results = {
        'config': {k: str(v) if isinstance(v, torch.device) else v 
                  for k, v in vars(config).items() if not k.startswith('__')},
        'feature_info': feature_info,
        'evaluation': evaluation_results,
        'fault_mapping': data_processor.fault_mapping
    }
    
    with open(os.path.join(config.output_dir, config.experiment_data_path), 'w') as f:
        json.dump(experiment_results, f, indent=4)
    
    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'loss_curve.png'))
    plt.close()
    
    logging.info(f"实验完成，结果保存在 {config.output_dir}")
    
    return evaluation_results


if __name__ == "__main__":
    # 设置随机种子，确保结果可复现
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        # 运行主函数
        results = main()
        logging.info("程序成功执行完毕")
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}", exc_info=True)
    
    