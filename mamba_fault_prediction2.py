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
from torch.utils.data import DataLoader, TensorDataset

# 1. 基于Mamba架构的故障预测模型
# 2. 完整的数据处理流程
# 3. 模型训练和评估功能
# 4. 结果可视化和保存
# 与原来的enhanced_fault_prediction.py相比，主要区别是：
#
# 1. 使用了model.py中的Mamba模型架构
# 2. 去掉了过采样部分，直接使用原始数据进行训练
# 3. 保留了多任务学习、焦点损失和Mixup数据增强等高级功能

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
    learning_rate: float = 1e-3  # 学习率
    weight_decay: float = 1e-4  # 权重衰减
    
    # 类别权重
    use_class_weights: bool = True
    # 针对TWF和RNF的额外权重倍数
    twf_weight_multiplier: float = 10.0  # TWF类别权重额外乘数
    rnf_weight_multiplier: float = 20.0  # RNF类别权重额外乘数
    
    # 多任务学习
    use_multitask: bool = True
    # 多任务损失权重
    multi_task_weight: float = 0.7  # 多分类任务权重
    binary_task_weight: float = 0.3  # 二分类任务权重
    
    # 焦点损失参数
    use_focal_loss: bool = True  # 使用焦点损失
    focal_gamma: float = 2.5  # 焦点损失gamma参数
    focal_alpha: float = 0.25  # 焦点损失alpha参数
    
    # 特征工程
    use_advanced_features: bool = True  # 使用高级特征工程
    
    # 数据增强
    use_mixup: bool = True  # 使用Mixup数据增强
    mixup_alpha: float = 0.2  # Mixup参数
    
    device: torch.device = None
    columns_to_drop: List[str] = None
    output_dir: str = None  # 将在 __post_init__ 中设置
    fault_types: List[str] = None  # 故障类型列表
    num_classes: int = 6  # 包括无故障和5种故障类型(HDF, PWF, OSF, TWF, RNF)

    def __post_init__(self):
        self.columns_to_drop = ['UDI', 'Product ID']  # 只删除不相关的ID列
        self.fault_types = ['HDF', 'PWF', 'OSF', 'TWF', 'RNF']  # 考虑所有5种故障类型

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
            vocab_size=num_classes,  # 这里不是真正用于词汇表，只是为了初始化
            d_state=config.d_state,
            expand=config.expand
        )
        
        # 特征嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Mamba层
        self.layers = nn.ModuleList([ResidualBlock(mamba_args) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model)
        
        # 分类头
        self.classifier = nn.Linear(config.d_model, num_classes)
        
        # 二分类头（用于多任务学习）
        if config.use_multitask:
            self.binary_classifier = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.LayerNorm(config.d_model // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, 2)  # 二分类：有故障/无故障
            )
            
        # 特定故障类型的专用分类器
        self.twf_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 2)  # 二分类：是否为TWF故障
        )
        
        self.rnf_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.LayerNorm(config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 2)  # 二分类：是否为RNF故障
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
        
        # TWF和RNF专用分类器
        twf_logits = self.twf_classifier(x)
        rnf_logits = self.rnf_classifier(x)
        
        # 如果使用多任务学习，返回多分类和二分类的结果
        if self.config.use_multitask:
            binary_logits = self.binary_classifier(x)
            return logits, binary_logits, twf_logits, rnf_logits
        else:
            return logits, twf_logits, rnf_logits


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
            
            # 添加TWF和RNF相关的特征组合
            df['TWF_feature'] = df['Torque [Nm]'] * df['Tool wear [min]'] / (df['Rotational speed [rpm]'] + 1e-8)
            df['RNF_feature'] = df['Rotational speed [rpm]'] * df['Process temperature [K]'] / (df['Tool wear [min]'] + 1e-8)
            
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
        columns_to_drop = self.config.columns_to_drop + ['Machine failure', 'Type', 'Has_Failure'] + self.config.fault_types
        df_processed = df.drop(columns=columns_to_drop, errors='ignore')
        
        # 分离特征和标签
        X = df_processed.drop(columns=['Fault_Type'])
        y = df['Fault_Type']
        
        # 计算类别权重
        class_counts = np.bincount(y)
        total_samples = len(y)
        class_weights = total_samples / (len(class_counts) * class_counts)
        
        # 对TWF和RNF类别增加额外权重
        twf_idx = list(self.fault_mapping.values()).index('TWF')
        rnf_idx = list(self.fault_mapping.values()).index('RNF')
        class_weights[twf_idx] *= self.config.twf_weight_multiplier
        class_weights[rnf_idx] *= self.config.rnf_weight_multiplier
        
        self.class_weights = torch.FloatTensor(class_weights).to(self.config.device)
        logging.info(f"类别权重: {class_weights}")
        
        # 数据集划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 转换为张量并移动到指定设备
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.config.device)
        y_train_tensor = torch.LongTensor(y_train.values if hasattr(y_train, 'values') else y_train).to(self.config.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.config.device)
        y_test_tensor = torch.LongTensor(y_test.values).to(self.config.device)
        
        # 创建二分类标签（有故障/无故障）
        y_train_binary = (y_train > 0).astype(int)
        y_train_binary_tensor = torch.LongTensor(y_train_binary if not hasattr(y_train_binary, 'values') else y_train_binary.values).to(self.config.device)
        
        # 创建TWF和RNF的二分类标签
        y_train_twf = (y_train == twf_idx).astype(int)
        y_train_twf_tensor = torch.LongTensor(y_train_twf if not hasattr(y_train_twf, 'values') else y_train_twf.values).to(self.config.device)
        
        y_train_rnf = (y_train == rnf_idx).astype(int)
        y_train_rnf_tensor = torch.LongTensor(y_train_rnf if not hasattr(y_train_rnf, 'values') else y_train_rnf.values).to(self.config.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, y_train_binary_tensor, y_train_twf_tensor, y_train_rnf_tensor)
        
        # 不使用加权采样器，直接创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True
        )
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True
        )
        
        # 记录特征信息
        feature_info = {
            'feature_names': X.columns.tolist(),
            'input_dim': X.shape[1],
            'num_classes': len(self.fault_mapping)
        }
        
        logging.info(f"特征维度: {feature_info['input_dim']}")
        logging.info(f"类别数量: {feature_info['num_classes']}")
        
        return train_loader, test_loader, X_test_tensor, y_test.values, feature_info


# Mixup数据增强
def mixup_data(x, y, alpha=0.2, device=None):
    """执行Mixup数据增强"""
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
        self.criterion_twf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0]).to(config.device))
        self.criterion_rnf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 20.0]).to(config.device))
        
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
        
        # 创建TWF和RNF的二分类验证标签
        twf_idx = list(train_loader.dataset.tensors[1].unique().cpu().numpy()).index(4)  # TWF索引
        rnf_idx = list(train_loader.dataset.tensors[1].unique().cpu().numpy()).index(5)  # RNF索引
        
        y_val_twf = (y_val == twf_idx).astype(int)
        y_val_twf_tensor = torch.LongTensor(y_val_twf).to(self.config.device)
        
        y_val_rnf = (y_val == rnf_idx).astype(int)
        y_val_rnf_tensor = torch.LongTensor(y_val_rnf).to(self.config.device)
        
        logging.info(f"开始训练，总轮次: {self.config.num_epochs}")
        
        for epoch in range(self.config.num_epochs):
            # 训练阶段
            self.model.train()
            epoch_loss = 0.0
            
            for batch_idx, (data, target, target_binary, target_twf, target_rnf) in enumerate(train_loader):
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
                        logits, binary_logits, twf_logits, rnf_logits = outputs
                        
                        # 计算多分类损失（使用Mixup）
                        loss_multi = mixup_criterion(
                            self.criterion_multi, logits, target_a, target_b, lam
                        )
                        
                        # 计算二分类损失（不使用Mixup）
                        loss_binary = self.criterion_binary(binary_logits, target_binary)
                        loss_twf = self.criterion_twf(twf_logits, target_twf)
                        loss_rnf = self.criterion_rnf(rnf_logits, target_rnf)
                        
                        # 总损失
                        loss = (self.config.multi_task_weight * loss_multi + 
                                self.config.binary_task_weight * loss_binary +
                                0.1 * loss_twf + 0.1 * loss_rnf)
                    else:
                        logits, twf_logits, rnf_logits = outputs
                        
                        # 计算多分类损失（使用Mixup）
                        loss_multi = mixup_criterion(
                            self.criterion_multi, logits, target_a, target_b, lam
                        )
                        
                        # 计算TWF和RNF专用分类器的损失
                        loss_twf = self.criterion_twf(twf_logits, target_twf)
                        loss_rnf = self.criterion_rnf(rnf_logits, target_rnf)
                        
                        # 总损失
                        loss = loss_multi + 0.1 * loss_twf + 0.1 * loss_rnf
                else:
                    # 前向传播
                    outputs = self.model(data)
                    
                    # 多任务学习
                    if self.config.use_multitask:
                        logits, binary_logits, twf_logits, rnf_logits = outputs
                        
                        # 计算多分类损失
                        loss_multi = self.criterion_multi(logits, target)
                        
                        # 计算二分类损失
                        loss_binary = self.criterion_binary(binary_logits, target_binary)
                        loss_twf = self.criterion_twf(twf_logits, target_twf)
                        loss_rnf = self.criterion_rnf(rnf_logits, target_rnf)
                        
                        # 总损失
                        loss = (self.config.multi_task_weight * loss_multi + 
                                self.config.binary_task_weight * loss_binary +
                                0.1 * loss_twf + 0.1 * loss_rnf)
                    else:
                        logits, twf_logits, rnf_logits = outputs
                        
                        # 计算多分类损失
                        loss_multi = self.criterion_multi(logits, target)
                        
                        # 计算TWF和RNF专用分类器的损失
                        loss_twf = self.criterion_twf(twf_logits, target_twf)
                        loss_rnf = self.criterion_rnf(rnf_logits, target_rnf)
                        
                        # 总损失
                        loss = loss_multi + 0.1 * loss_twf + 0.1 * loss_rnf
                
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # 计算平均训练损失
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                # 前向传播
                outputs = self.model(X_val)
                
                # 多任务学习
                if self.config.use_multitask:
                    logits, binary_logits, twf_logits, rnf_logits = outputs
                    
                    # 计算多分类损失
                    loss_multi = self.criterion_multi(logits, y_val_tensor)
                    
                    # 计算二分类损失
                    loss_binary = self.criterion_binary(binary_logits, y_val_binary_tensor)
                    loss_twf = self.criterion_twf(twf_logits, y_val_twf_tensor)
                    loss_rnf = self.criterion_rnf(rnf_logits, y_val_rnf_tensor)
                    
                    # 总损失
                    val_loss = (self.config.multi_task_weight * loss_multi + 
                                self.config.binary_task_weight * loss_binary +
                                0.1 * loss_twf + 0.1 * loss_rnf)
                else:
                    logits, twf_logits, rnf_logits = outputs
                    
                    # 计算多分类损失
                    loss_multi = self.criterion_multi(logits, y_val_tensor)
                    
                    # 计算TWF和RNF专用分类器的损失
                    loss_twf = self.criterion_twf(twf_logits, y_val_twf_tensor)
                    loss_rnf = self.criterion_rnf(rnf_logits, y_val_rnf_tensor)
                    
                    # 总损失
                    val_loss = loss_multi + 0.1 * loss_twf + 0.1 * loss_rnf
            
            val_losses.append(val_loss.item())
            
            # 更新学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info(f"当前学习率: {current_lr:.6f}")
            
            # 打印训练信息
            logging.info(f"轮次 {epoch+1}/{self.config.num_epochs}, 训练损失: {avg_train_loss:.4f}, 验证损失: {val_loss:.4f}")
            
            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # 保存最佳模型
                torch.save(self.model.state_dict(), self.config.model_save_path)
                logging.info(f"最佳模型已保存到 {self.config.model_save_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    logging.info(f"早停触发，共训练 {epoch+1} 轮")
                    break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(self.config.model_save_path))
        logging.info("加载最佳模型完成")
        
        return train_losses, val_losses


class MambaModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config: MambaFaultConfig):
        self.config = config
    
    def evaluate(self, model, X_test, y_test, fault_mapping):
        """评估模型性能"""
        model.eval()
        
        with torch.no_grad():
            # 前向传播
            outputs = model(X_test)
            
            # 获取预测结果
            if self.config.use_multitask:
                logits, binary_logits, twf_logits, rnf_logits = outputs
            else:
                logits, twf_logits, rnf_logits = outputs
            
            # 获取预测类别和概率
            y_probs = F.softmax(logits, dim=1).cpu().numpy()
            y_pred = np.argmax(y_probs, axis=1)
            
            # 获取二分类预测
            if self.config.use_multitask:
                binary_probs = F.softmax(binary_logits, dim=1).cpu().numpy()
                binary_pred = np.argmax(binary_probs, axis=1)
            else:
                binary_pred = (y_pred > 0).astype(int)
            
            # 获取TWF和RNF的专用预测
            twf_probs = F.softmax(twf_logits, dim=1).cpu().numpy()
            twf_pred = np.argmax(twf_probs, axis=1)
            
            rnf_probs = F.softmax(rnf_logits, dim=1).cpu().numpy()
            rnf_pred = np.argmax(rnf_probs, axis=1)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        logging.info(f"准确率: {accuracy:.4f}")
        logging.info(f"F1分数(宏平均): {f1_macro:.4f}")
        logging.info(f"F1分数(加权平均): {f1_weighted:.4f}")
        
        # 生成分类报告
        class_names = [fault_mapping[i] for i in range(len(fault_mapping))]
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        logging.info(f"分类报告:\n{classification_report(y_test, y_pred, target_names=class_names, zero_division=0)}")
        
        # 生成混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 创建二分类标签（有故障/无故障）
        y_test_binary = (y_test > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
        
        # 计算二分类的概率（所有故障类型的概率之和）
        y_prob_binary = np.sum(y_probs[:, 1:], axis=1)
        
        # 二分类评估指标
        binary_accuracy = accuracy_score(y_test_binary, y_pred_binary)
        binary_f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
        binary_precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
        binary_recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
        
        logging.info(f"二分类准确率: {binary_accuracy:.4f}")
        logging.info(f"二分类F1分数: {binary_f1:.4f}")
        logging.info(f"二分类精确率: {binary_precision:.4f}")
        logging.info(f"二分类召回率: {binary_recall:.4f}")
        
        # 评估TWF和RNF专用分类器
        twf_idx = list(fault_mapping.values()).index('TWF')
        rnf_idx = list(fault_mapping.values()).index('RNF')
        
        y_test_twf = (y_test == twf_idx).astype(int)
        y_test_rnf = (y_test == rnf_idx).astype(int)
        
        # TWF专用分类器评估
        twf_accuracy = accuracy_score(y_test_twf, twf_pred)
        twf_f1 = f1_score(y_test_twf, twf_pred, zero_division=0)
        twf_precision = precision_score(y_test_twf, twf_pred, zero_division=0)
        twf_recall = recall_score(y_test_twf, twf_pred, zero_division=0)
        
        logging.info(f"TWF专用分类器准确率: {twf_accuracy:.4f}")
        logging.info(f"TWF专用分类器F1分数: {twf_f1:.4f}")
        logging.info(f"TWF专用分类器精确率: {twf_precision:.4f}")
        logging.info(f"TWF专用分类器召回率: {twf_recall:.4f}")
        
        # RNF专用分类器评估
        rnf_accuracy = accuracy_score(y_test_rnf, rnf_pred)
        rnf_f1 = f1_score(y_test_rnf, rnf_pred, zero_division=0)
        rnf_precision = precision_score(y_test_rnf, rnf_pred, zero_division=0)
        rnf_recall = recall_score(y_test_rnf, rnf_pred, zero_division=0)
        
        logging.info(f"RNF专用分类器准确率: {rnf_accuracy:.4f}")
        logging.info(f"RNF专用分类器F1分数: {rnf_f1:.4f}")
        logging.info(f"RNF专用分类器精确率: {rnf_precision:.4f}")
        logging.info(f"RNF专用分类器召回率: {rnf_recall:.4f}")
        
        # 创建评估结果字典
        evaluation_results = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'binary_metrics': {
                'accuracy': float(binary_accuracy),
                'f1': float(binary_f1),
                'precision': float(binary_precision),
                'recall': float(binary_recall)
            },
            'twf_metrics': {
                'accuracy': float(twf_accuracy),
                'f1': float(twf_f1),
                'precision': float(twf_precision),
                'recall': float(twf_recall)
            },
            'rnf_metrics': {
                'accuracy': float(rnf_accuracy),
                'f1': float(rnf_f1),
                'precision': float(rnf_precision),
                'recall': float(rnf_recall)
            }
        }
        
        # 绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, "confusion_matrix.svg"), format='svg', bbox_inches='tight')
        plt.close()
        
        # 绘制二分类混淆矩阵
        binary_class_names = ['无故障', '有故障']
        plt.figure(figsize=(8, 6))
        cm_binary = confusion_matrix(y_test_binary, y_pred_binary)
        sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues',
                   xticklabels=binary_class_names,
                   yticklabels=binary_class_names)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('二分类混淆矩阵')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, "binary_confusion_matrix.svg"), format='svg', bbox_inches='tight')
        plt.close()
        
        # 绘制ROC曲线
        plt.figure(figsize=(12, 10))
        
        # 多分类ROC曲线
        for i in range(len(fault_mapping)):
            if i == 0:  # 无故障类别
                label = f'{fault_mapping[i]} vs 其他'
            else:
                label = f'{fault_mapping[i]} vs 其他'
            
            # 计算当前类别的ROC曲线
            y_test_binary = (y_test == i).astype(int)
            y_score = y_probs[:, i]
            
            fpr, tpr, _ = roc_curve(y_test_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
        
        # 二分类ROC曲线
        fpr_binary, tpr_binary, _ = roc_curve(y_test_binary, y_prob_binary)
        roc_auc_binary = auc(fpr_binary, tpr_binary)
        plt.plot(fpr_binary, tpr_binary, lw=2, label=f'有故障 vs 无故障 (AUC = {roc_auc_binary:.2f})', linestyle='--')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('接收者操作特征曲线 (ROC)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, "roc_curves.svg"), format='svg', bbox_inches='tight')
        plt.close()
        
        # 绘制精确率-召回率曲线
        plt.figure(figsize=(12, 10))
        
        # 多分类PR曲线
        for i in range(len(fault_mapping)):
            if i == 0:  # 无故障类别
                continue  # 跳过无故障类别，因为我们更关注故障检测
            
            # 计算当前类别的PR曲线
            y_test_binary = (y_test == i).astype(int)
            y_score = y_probs[:, i]
            
            precision, recall, _ = precision_recall_curve(y_test_binary, y_score)
            avg_precision = average_precision_score(y_test_binary, y_score)
            
            plt.plot(recall, precision, lw=2, label=f'{fault_mapping[i]} (AP = {avg_precision:.2f})')
        
        # 二分类PR曲线
        precision_binary, recall_binary, _ = precision_recall_curve(y_test_binary, y_prob_binary)
        avg_precision_binary = average_precision_score(y_test_binary, y_prob_binary)
        plt.plot(recall_binary, precision_binary, lw=2, label=f'有故障 vs 无故障 (AP = {avg_precision_binary:.2f})', linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, "pr_curves.svg"), format='svg', bbox_inches='tight')
        plt.close()
        
        # 保存评估结果到JSON文件
        evaluation_results_path = os.path.join(self.config.output_dir, self.config.experiment_data_path)
        with open(evaluation_results_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
        
        logging.info(f"评估结果已保存到 {evaluation_results_path}")
        
        return evaluation_results
    
    def visualize_training_history(self, train_losses, val_losses):
        """可视化训练历史"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, "loss_history.svg"), format='svg', bbox_inches='tight')
        plt.close()
        
        logging.info(f"训练历史图表已保存到 {os.path.join(self.config.output_dir, 'loss_history.svg')}")

def main():
    """主函数"""
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 初始化配置
    config = MambaFaultConfig()
    logging.info(f"使用设备: {config.device}")
    
    # 数据处理
    data_processor = MambaDataProcessor(config)
    train_loader, test_loader, X_test, y_test, feature_info = data_processor.load_and_preprocess()
    
    # 创建模型
    model = MambaFaultPredictionModel(
        config=config,
        input_dim=feature_info['input_dim'],
        num_classes=feature_info['num_classes']
    ).to(config.device)
    
    # 打印模型结构
    logging.info(f"模型结构:\n{model}")
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"总参数数量: {total_params:,}")
    logging.info(f"可训练参数数量: {trainable_params:,}")
    
    # 训练模型
    trainer = MambaModelTrainer(config, model, data_processor.class_weights)
    train_losses, val_losses = trainer.train(train_loader, X_test, y_test)
    
    # 评估模型
    evaluator = MambaModelEvaluator(config)
    evaluation_results = evaluator.evaluate(model, X_test, y_test, data_processor.fault_mapping)
    
    # 可视化训练历史
    evaluator.visualize_training_history(train_losses, val_losses)
    
    # 打印最终结果
    logging.info("训练和评估完成！")
    logging.info(f"最终准确率: {evaluation_results['accuracy']:.4f}")
    logging.info(f"最终F1分数(宏平均): {evaluation_results['f1_macro']:.4f}")
    logging.info(f"最终F1分数(加权平均): {evaluation_results['f1_weighted']:.4f}")
    logging.info(f"二分类F1分数: {evaluation_results['binary_metrics']['f1']:.4f}")
    logging.info(f"TWF专用分类器F1分数: {evaluation_results['twf_metrics']['f1']:.4f}")
    logging.info(f"RNF专用分类器F1分数: {evaluation_results['rnf_metrics']['f1']:.4f}")
    
    # 保存实验配置
    config_dict = {k: str(v) if isinstance(v, (torch.device, type)) else v 
                  for k, v in config.__dict__.items() 
                  if not k.startswith('__') and not callable(v)}
    
    # 移除不可序列化的对象
    if 'device' in config_dict:
        config_dict['device'] = str(config_dict['device'])
    if 'class_weights' in config_dict:
        config_dict['class_weights'] = None
    
    config_path = os.path.join(config.output_dir, "experiment_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)
    
    logging.info(f"实验配置已保存到 {config_path}")
    
    return model, evaluation_results



if __name__ == "__main__":
    main()