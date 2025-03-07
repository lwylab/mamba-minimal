"""
基于Mamba状态空间模型的故障预测系统
使用真正的Mamba架构进行时序数据建模和故障预测
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional, Union

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

# 导入Mamba模型组件
from model import ModelArgs, Mamba, ResidualBlock, MambaBlock, RMSNorm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mamba_training_v2.log'),
        logging.StreamHandler()
    ]
)

# 设置 Matplotlib 的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义焦点损失函数
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


# Mixup数据增强
def mixup_data(x, y, alpha=1.0, device='cpu'):
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
    """计算Mixup损失"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


@dataclass
class MambaConfig:
    """Mamba模型配置"""
    # 数据路径
    data_path: str = "./ai4i2020.csv"
    model_save_path: str = "best_mamba_model_v2.pth"
    output_dir: str = "mamba_results"
    
    # 训练参数
    batch_size: int = 32
    num_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    early_stopping_patience: int = 20
    
    # Mamba模型参数
    d_model: int = 128
    n_layer: int = 4
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    
    # 数据处理参数
    test_size: float = 0.2
    random_state: int = 42
    use_oversampling: bool = True
    use_class_weights: bool = True
    
    # 损失函数参数
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # 数据增强
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    
    # 设备
    device: torch.device = None
    
    def __post_init__(self):
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{self.output_dir}_{timestamp}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 更新模型保存路径
        self.model_save_path = os.path.join(self.output_dir, self.model_save_path)
        
        logging.info(f"使用设备: {self.device}")
        logging.info(f"输出目录: {self.output_dir}")


class MambaSequenceModel(nn.Module):
    """基于Mamba的序列模型"""
    def __init__(self, input_dim: int, num_classes: int, config: MambaConfig):
        super().__init__()
        self.config = config
        
        # 创建Mamba模型参数
        self.mamba_args = ModelArgs(
            d_model=config.d_model,
            n_layer=config.n_layer,
            vocab_size=num_classes,  # 这里不是真的词汇表大小，只是为了兼容ModelArgs
            d_state=config.d_state,
            expand=config.expand,
            dt_rank=config.dt_rank
        )
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, config.d_model)
        
        # Mamba层
        self.layers = nn.ModuleList([
            ResidualBlock(self.mamba_args) for _ in range(config.n_layer)
        ])
        
        # 最终层归一化
        self.norm_f = RMSNorm(config.d_model)
        
        # 分类头
        self.classifier = nn.Linear(config.d_model, num_classes)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
        Returns:
            logits: 分类logits [batch_size, num_classes]
        """
        # 投影到模型维度
        x = self.input_proj(x)
        
        # 通过Mamba层
        for layer in self.layers:
            x = layer(x)
        
        # 最终归一化
        x = self.norm_f(x)
        
        # 取序列的最后一个时间步进行分类
        x = x[:, -1]
        
        # 分类
        logits = self.classifier(x)
        
        return logits


class DataProcessor:
    """数据处理类"""
    def __init__(self, config: MambaConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.fault_mapping = None
        self.class_weights = None
    
    def load_data(self):
        """加载并预处理数据"""
        logging.info(f"加载数据: {self.config.data_path}")
        df = pd.read_csv(self.config.data_path)
        
        # 数据探索
        logging.info(f"数据集形状: {df.shape}")
        logging.info(f"数据集列: {df.columns.tolist()}")
        
        # 特征工程
        df = self._feature_engineering(df)
        
        # 创建故障类型标签
        df = self._create_fault_labels(df)
        
        # 提取特征和标签
        X, y = self._extract_features_labels(df)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state, stratify=y
        )
        
        # 标准化特征
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # 计算类别权重
        self._compute_class_weights(y_train)
        
        # 过采样少数类
        if self.config.use_oversampling:
            X_train, y_train = self._oversample(X_train, y_train)
        
        # 转换为序列格式
        X_train_seq = self._to_sequence(X_train)
        X_test_seq = self._to_sequence(X_test)
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.config.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.config.device)
        X_test_tensor = torch.FloatTensor(X_test_seq).to(self.config.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.config.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        
        return train_loader, X_test_tensor, y_test, self.fault_mapping
    
    def _feature_engineering(self, df):
        """特征工程"""
        # 编码产品类型
        df['Type_encoded'] = self.label_encoder.fit_transform(df['Type'])
        
        # 添加高级特征
        # 温度差
        df['Temp_diff'] = df['Air temperature [K]'] - df['Process temperature [K]']
        
        # 旋转力矩比
        df['Torque_Rotational_ratio'] = df['Torque [Nm]'] / (df['Rotational speed [rpm]'] + 1e-8)
        
        # 功率特征
        df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']
        
        # 工具磨损率
        df['Tool_wear_rate'] = df['Tool wear [min]'] / (df['Process temperature [K]'] + 1e-8)
        
        # 交互特征
        df['Temp_Tool_interaction'] = df['Process temperature [K]'] * df['Tool wear [min]']
        df['Rotation_Tool_interaction'] = df['Rotational speed [rpm]'] * df['Tool wear [min]']
        df['Torque_Tool_interaction'] = df['Torque [Nm]'] * df['Tool wear [min]']
        
        # 非线性特征
        df['Rotational_speed_squared'] = df['Rotational speed [rpm]'] ** 2
        df['Tool_wear_squared'] = df['Tool wear [min]'] ** 2
        df['Torque_squared'] = df['Torque [Nm]'] ** 2
        
        logging.info("特征工程完成")
        return df
    
    def _create_fault_labels(self, df):
        """创建故障类型标签"""
        # 故障类型
        fault_types = ['HDF', 'PWF', 'OSF', 'TWF', 'RNF']
        
        # 创建故障类型标签
        df['Fault_Type'] = 0  # 默认为无故障
        
        # 为每种故障类型分配标签
        for i, fault_type in enumerate(fault_types, 1):
            mask = df[fault_type] == 1
            df.loc[mask, 'Fault_Type'] = i
        
        # 创建故障类型映射
        self.fault_mapping = {0: '无故障'}
        for i, fault_type in enumerate(fault_types, 1):
            self.fault_mapping[i] = fault_type
        
        logging.info(f"故障类型映射: {self.fault_mapping}")
        logging.info(f"故障类型分布:\n{df['Fault_Type'].value_counts().sort_index()}")
        
        return df
    
    def _extract_features_labels(self, df):
        """提取特征和标签"""
        # 删除不需要的列
        columns_to_drop = ['UDI', 'Product ID', 'Type', 'Machine failure', 'HDF', 'PWF', 'OSF', 'TWF', 'RNF']
        
        # 提取特征和标签
        X = df.drop(columns=columns_to_drop + ['Fault_Type']).copy()
        y = df['Fault_Type'].values
        
        logging.info(f"特征数量: {X.shape[1]}")
        logging.info(f"特征列表: {X.columns.tolist()}")
        
        return X, y
    
    def _compute_class_weights(self, y):
        """计算类别权重"""
        class_counts = np.bincount(y)
        total_samples = len(y)
        
        # 计算类别权重
        self.class_weights = torch.FloatTensor(
            total_samples / (len(class_counts) * class_counts)
        ).to(self.config.device)
        
        # 调整无故障类别的权重
        self.class_weights[0] *= 1.5
        
        logging.info(f"类别权重: {self.class_weights}")
    
    def _oversample(self, X, y):
        """对少数类进行过采样"""
        logging.info("执行过采样...")
        
        # 获取各类别样本数
        class_counts = np.bincount(y)
        majority_class = 0  # 无故障类别
        majority_count = class_counts[majority_class]
        
        # 合并特征和标签
        data = np.column_stack((X, y))
        
        # 分离各类别数据
        class_data = {}
        for i in range(len(class_counts)):
            class_data[i] = data[data[:, -1] == i]
        
        # 过采样少数类
        oversampled_data = [class_data[majority_class]]
        for i in range(1, len(class_counts)):
            minority_count = class_counts[i]
            target_count = int(majority_count * 0.3)  # 设置为多数类的30%
            
            # 确保至少是原来的3倍
            target_count = max(target_count, minority_count * 3)
            
            # 过采样
            if minority_count < target_count:
                oversampled = resample(
                    class_data[i],
                    replace=True,
                    n_samples=target_count,
                    random_state=self.config.random_state
                )
                oversampled_data.append(oversampled)
            else:
                oversampled_data.append(class_data[i])
        
        # 合并过采样后的数据
        oversampled_data = np.vstack(oversampled_data)
        
        # 分离特征和标签
        X_oversampled = oversampled_data[:, :-1]
        y_oversampled = oversampled_data[:, -1].astype(int)
        
        logging.info(f"过采样前样本分布: {np.bincount(y)}")
        logging.info(f"过采样后样本分布: {np.bincount(y_oversampled)}")
        
        return X_oversampled, y_oversampled
    
    def _to_sequence(self, X):
        """将特征转换为序列格式"""
        # 对于非时序数据，我们创建一个人工序列
        # 这里我们将每个样本复制5次，形成一个长度为5的序列
        seq_len = 5
        batch_size, feature_dim = X.shape
        
        # 创建序列
        X_seq = np.zeros((batch_size, seq_len, feature_dim))
        
        for i in range(batch_size):
            # 基础特征
            base_features = X[i]
            
            # 创建序列变化
            for j in range(seq_len):
                # 添加微小的随机变化，模拟时序数据
                noise = np.random.normal(0, 0.01, feature_dim)
                X_seq[i, j] = base_features + noise * j
        
        return X_seq


class ModelTrainer:
    """模型训练器"""
    def __init__(self, model, config: MambaConfig, class_weights=None):
        self.model = model
        self.config = config
        self.class_weights = class_weights
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )
        
        # 损失函数
        if config.use_focal_loss:
            self.criterion = FocalLoss(
                alpha=config.focal_alpha,
                gamma=config.focal_gamma,
                weight=class_weights if config.use_class_weights else None
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights if config.use_class_weights else None
            )
        
        # 早停设置
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logging.info("模型训练器初始化完成")
    
    def train(self, train_loader, X_val, y_val):
        """训练模型"""
        train_losses = []
        val_losses = []
        
        # 将验证标签转换为张量
        y_val_tensor = torch.LongTensor(y_val).to(self.config.device)
        
        logging.info(f"开始训练，总轮次: {self.config.num_epochs}")
        
        for epoch in range(self.config.num_epochs):
            # 训练阶段
            self.model.train()
            epoch_loss = 0.0
            
            # 添加进度跟踪
            batch_count = len(train_loader)
            
            for batch_idx, (data, target) in enumerate(train_loader):
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
                    logits = self.model(data)
                    
                    # 计算损失（使用Mixup）
                    loss = mixup_criterion(
                        self.criterion, logits, target_a, target_b, lam
                    )
                else:
                    # 前向传播
                    logits = self.model(data)
                    
                    # 计算损失
                    loss = self.criterion(logits, target)
                
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # 计算平均训练损失
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            val_loss, val_metrics = self.evaluate(X_val, y_val, y_val_tensor)
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
    
    def evaluate(self, X, y, y_tensor=None):
        """评估模型"""
        self.model.eval()
        
        if y_tensor is None:
            y_tensor = torch.LongTensor(y).to(self.config.device)
        
        with torch.no_grad():
            # 前向传播
            logits = self.model(X)
            
            # 计算损失
            loss = self.criterion(logits, y_tensor)
            
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
            logits = self.model(X)
            
            # 获取预测结果
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            return preds.cpu().numpy(), probs.cpu().numpy()


class ModelEvaluator:
    """模型评估器"""
    def __init__(self, config: MambaConfig, fault_mapping: dict):
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
            json.dump(evaluation_results, f, indent=4, cls=NumpyEncoder)
        
        return evaluation_results


# 用于JSON序列化NumPy数据类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def main():
    """主函数"""
    # 设置随机种子，确保结果可复现
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 创建配置
    config = MambaConfig()
    
    # 创建数据处理器
    data_processor = DataProcessor(config)
    
    # 加载和预处理数据
    train_loader, X_test, y_test, fault_mapping = data_processor.load_data()
    
    # 获取输入维度和类别数
    input_dim = X_test.shape[2]  # [batch_size, seq_len, input_dim]
    num_classes = len(fault_mapping)
    
    # 创建模型
    model = MambaSequenceModel(
        input_dim=input_dim,
        num_classes=num_classes,
        config=config
    ).to(config.device)
    
    # 打印模型结构
    logging.info(f"模型结构:\n{model}")
    
    # 创建训练器
    trainer = ModelTrainer(
        model=model,
        config=config,
        class_weights=data_processor.class_weights
    )
    
    # 训练模型
    train_losses, val_losses = trainer.train(train_loader, X_test, y_test)
    
    # 获取预测结果
    y_pred, y_prob = trainer.predict(X_test)
    
    # 创建评估器
    evaluator = ModelEvaluator(config, fault_mapping)
    
    # 评估和可视化结果
    evaluation_results = evaluator.evaluate_and_visualize(y_test, y_pred, y_prob)
    
    # 保存实验结果到JSON文件
    experiment_results = {
        'config': {k: str(v) if isinstance(v, torch.device) else v 
                  for k, v in vars(config).items() if not k.startswith('__')},
        'evaluation': evaluation_results,
        'fault_mapping': fault_mapping
    }
    
    with open(os.path.join(config.output_dir, 'experiment_results.json'), 'w') as f:
        json.dump(experiment_results, f, indent=4, cls=NumpyEncoder)
    
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
    try:
        # 运行主函数
        results = main()
        logging.info("程序成功执行完毕")
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}", exc_info=True)

# - Mamba状态空间模型进行序列建模
# - 完整的数据处理流程，包括特征工程和过采样
# - 高级训练技术，如Focal Loss和Mixup数据增强
# - 全面的模型评估和可视化功能
# - 详细的日志记录和结果保存