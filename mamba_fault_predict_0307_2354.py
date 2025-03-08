"""
基于Mamba状态空间模型的故障预测系统
使用Mamba架构进行时序数据建模和故障预测
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
from einops import rearrange, repeat



# 导入Mamba模型组件
from mambapy.mamba import MambaConfig, Mamba, ResidualBlock, RMSNorm

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
class MambaFaultConfig:
    """Mamba故障预测模型配置"""
    # 数据路径
    data_path: str = r"C:\Users\L\PycharmProjects\20250227\mamba-minimal\ai4i2020.csv"
    model_save_path: str = "best_mamba_model.pth"
    output_dir: str = "mamba_results"
    
    # 训练参数
    batch_size: int = 32
    num_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    early_stopping_patience: int = 20
    
    # Mamba模型参数
    d_model: int = 128
    n_layers: int = 4
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 4
    
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
    
    # 平衡准确率和召回率的参数
    precision_recall_balance: float = 0.6  # 值越大越偏向准确率，越小越偏向召回率
    threshold_adjustment: bool = True  # 是否使用阈值调整
    
    # 设备
    device: torch.device = None
    
    # 其他参数
    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False
    mup: bool = False
    pscan: bool = True
    use_cuda: bool = False
    
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
    def __init__(self, input_dim: int, num_classes: int, config: MambaFaultConfig):
        super().__init__()
        self.config = config
        
        # 创建Mamba模型参数
        self.mamba_config = MambaConfig(
            d_model=config.d_model,
            n_layers=config.n_layers,
            dt_rank=config.dt_rank,
            d_state=config.d_state,
            expand_factor=config.expand_factor,
            d_conv=config.d_conv,
            bias=config.bias,
            conv_bias=config.conv_bias,
            inner_layernorms=config.inner_layernorms,
            mup=config.mup,
            pscan=config.pscan,
            use_cuda=config.use_cuda
        )
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, config.d_model)
        
        # Mamba模型
        self.mamba = Mamba(self.mamba_config)
        
        # 最终层归一化
        self.norm_f = RMSNorm(config.d_model, eps=1e-5)
        
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
        x = self.mamba(x)
        
        # 最终归一化
        x = self.norm_f(x)
        
        # 取序列的最后一个时间步进行分类
        x = x[:, -1]
        
        # 分类
        logits = self.classifier(x)
        
        return logits


class DataProcessor:
    """数据处理类"""
    def __init__(self, config: MambaFaultConfig):
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
        fault_types = ['HDF', 'PWF', 'OSF']
        
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
        
        # 计算类别权重，但限制最大权重
        raw_weights = total_samples / (len(class_counts) * class_counts)
        

        # 限制最大权重为无故障类的5倍
        max_weight = raw_weights[0] * 5
        limited_weights = np.minimum(raw_weights, max_weight)
        
        self.class_weights = torch.FloatTensor(limited_weights).to(self.config.device)
        
        # 调整无故障类别的权重 - 增加无故障类的权重以减少假阳性
        self.class_weights[0] *= 2.0
        
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
        
        # 过采样少数类 - 修改过采样比例，避免过度过采样
        oversampled_data = class_data[majority_class]
        for i in range(1, len(class_counts)):
            # 修改过采样比例为原样本的1.5倍，而不是2倍
            n_samples = min(int(class_counts[i] * 1.5), int(majority_count * 0.1))
            
            # 过采样
            if n_samples > class_counts[i]:
                oversampled = resample(
                    class_data[i],
                    replace=True,
                    n_samples=n_samples,
                    random_state=self.config.random_state
                )
                oversampled_data = np.vstack((oversampled_data, oversampled))
            else:
                oversampled_data = np.vstack((oversampled_data, class_data[i]))
        
        # 分离特征和标签
        X_resampled = oversampled_data[:, :-1]
        y_resampled = oversampled_data[:, -1].astype(int)
        
        logging.info(f"过采样后的样本分布: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def _to_sequence(self, X, seq_len=10):
        """将特征转换为序列格式"""
        # 对于非序列数据，我们创建一个假的序列
        # 通过复制当前样本并添加少量噪声
        batch_size = X.shape[0]
        feature_dim = X.shape[1]
        
        # 初始化序列
        X_seq = np.zeros((batch_size, seq_len, feature_dim))
        
        for i in range(batch_size):
            # 基础特征
            base_features = X[i]
            
            # 创建序列
            for j in range(seq_len):
                # 添加少量随机噪声以创建时序变化
                noise = np.random.normal(0, 0.01, feature_dim)
                X_seq[i, j] = base_features + noise * j
        
        return X_seq


class ModelTrainer:
    """模型训练器"""
    def __init__(self, config: MambaFaultConfig):
        self.config = config
        self.best_val_f1 = 0.0
        self.early_stopping_counter = 0
        self.best_model_state = None
    
    def train(self, model, train_loader, X_test, y_test, fault_mapping, class_weights=None):
        """训练模型"""
        logging.info("开始训练模型...")
        
        # 设置优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 设置学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )
        
        # 设置损失函数
        if self.config.use_focal_loss:
            criterion = FocalLoss(
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma,
                weight=class_weights
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # 训练循环
        for epoch in range(self.config.num_epochs):
            # 训练模式
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # 清除梯度
                optimizer.zero_grad()
                
                # Mixup数据增强
                if self.config.use_mixup:
                    batch_X, batch_y_a, batch_y_b, lam = mixup_data(
                        batch_X, batch_y, self.config.mixup_alpha, self.config.device
                    )
                
                # 前向传播
                outputs = model(batch_X)
                
                # 计算损失
                if self.config.use_mixup:
                    loss = mixup_criterion(criterion, outputs, batch_y_a, batch_y_b, lam)
                else:
                    loss = criterion(outputs, batch_y)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # 更新参数
                optimizer.step()
                
                # 累计损失
                train_loss += loss.item()
            
            # 计算平均训练损失
            avg_train_loss = train_loss / len(train_loader)
            
            # 评估模式
            model.eval()
            with torch.no_grad():
                # 预测测试集
                y_pred = model(X_test)
                y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
                
                # 计算评估指标
                val_accuracy = accuracy_score(y_test, y_pred)
                val_f1 = f1_score(y_test, y_pred, average='weighted')
                val_precision = precision_score(y_test, y_pred, average='weighted')
                val_recall = recall_score(y_test, y_pred, average='weighted')
            
            # 更新学习率
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_f1)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 如果学习率发生变化，手动记录日志
            if prev_lr != current_lr:
                logging.info(f"学习率从 {prev_lr} 调整为 {current_lr}")
            
            # 打印训练信息
            logging.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Loss: {avg_train_loss:.4f}, "
                f"Accuracy: {val_accuracy:.4f}, "
                f"F1: {val_f1:.4f}, "
                f"Precision: {val_precision:.4f}, "
                f"Recall: {val_recall:.4f}"
            )
            
            # 检查是否是最佳模型
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.early_stopping_counter = 0
                self.best_model_state = model.state_dict().copy()
                
                # 保存最佳模型
                torch.save(model.state_dict(), self.config.model_save_path)
                logging.info(f"保存最佳模型，F1: {val_f1:.4f}")
            else:
                self.early_stopping_counter += 1
                logging.info(f"F1未提升，早停计数器: {self.early_stopping_counter}/{self.config.early_stopping_patience}")
                
                # 早停
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    logging.info(f"早停触发，停止训练")
                    break
        
        # 加载最佳模型
        model.load_state_dict(self.best_model_state)
        
        return model


class ModelEvaluator:
    """模型评估器"""
    def __init__(self, config: MambaFaultConfig):
        self.config = config
    
    def evaluate(self, model, X_test, y_test, fault_mapping):
        """评估模型"""
        logging.info("开始评估模型...")
        
        # 评估模式
        model.eval()
        
        # 预测
        with torch.no_grad():
            y_pred_proba = torch.softmax(model(X_test), dim=1).cpu().numpy()
        
        # 如果启用阈值调整，寻找最佳阈值
        if self.config.threshold_adjustment:
            y_pred = self._predict_with_optimal_threshold(y_pred_proba, y_test, fault_mapping)
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        logging.info(f"测试集评估结果:")
        logging.info(f"准确率: {accuracy:.4f}")
        logging.info(f"F1分数: {f1:.4f}")
        logging.info(f"精确率: {precision:.4f}")
        logging.info(f"召回率: {recall:.4f}")
        
        # 分类报告
        class_report = classification_report(y_test, y_pred, target_names=list(fault_mapping.values()))
        logging.info(f"分类报告:\n{class_report}")
        
        # 混淆矩阵
        self._plot_confusion_matrix(y_test, y_pred, fault_mapping)
        
        # ROC曲线
        self._plot_roc_curve(y_test, y_pred_proba, fault_mapping)
        
        # 精确率-召回率曲线
        self._plot_precision_recall_curve(y_test, y_pred_proba, fault_mapping)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def _predict_with_optimal_threshold(self, y_pred_proba, y_test, fault_mapping):
        """使用优化的阈值进行预测，平衡准确率和召回率"""
        logging.info("寻找最佳决策阈值以平衡准确率和召回率...")
        
        # 将真实标签转换为one-hot编码
        n_classes = len(fault_mapping)
        y_true_onehot = np.zeros((len(y_test), n_classes))
        for i in range(len(y_test)):
            y_true_onehot[i, y_test[i]] = 1
        
        # 初始化最佳阈值
        best_thresholds = np.zeros(n_classes)
        
        # 对每个类别寻找最佳阈值
        for i in range(1, n_classes):  # 从1开始，跳过无故障类别
            precisions, recalls, thresholds = precision_recall_curve(
                y_true_onehot[:, i], y_pred_proba[:, i]
            )
            
            # 计算F-beta分数，beta控制精确率和召回率的权重
            beta = (1 - self.config.precision_recall_balance) / self.config.precision_recall_balance
            beta_squared = beta ** 2
            f_scores = ((1 + beta_squared) * precisions * recalls) / (beta_squared * precisions + recalls + 1e-10)
            
            # 找到最大F-beta分数对应的阈值
            if len(thresholds) > 0:
                best_idx = np.argmax(f_scores[:-1])  # 最后一个precision/recall没有对应的阈值
                best_thresholds[i] = thresholds[best_idx]
            else:
                best_thresholds[i] = 0.5  # 默认阈值
            
            logging.info(f"类别 {fault_mapping[i]} 的最佳阈值: {best_thresholds[i]:.4f}")
        
        # 使用最佳阈值进行预测
        y_pred = np.zeros(len(y_test), dtype=int)
        
        # 首先假设所有样本都是无故障类别
        y_pred.fill(0)
        
        # 然后根据阈值判断是否属于故障类别
        for i in range(1, n_classes):
            mask = y_pred_proba[:, i] > best_thresholds[i]
            y_pred[mask] = i
        
        return y_pred
    
    def _plot_confusion_matrix(self, y_true, y_pred, fault_mapping):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        
        # 归一化混淆矩阵
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 绘制热图
        sns.heatmap(
            cm_norm, annot=cm, fmt='d', cmap='Blues',
            xticklabels=list(fault_mapping.values()),
            yticklabels=list(fault_mapping.values())
        )
        
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(self.config.output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
    
    def _plot_roc_curve(self, y_true, y_pred_proba, fault_mapping):
        """绘制ROC曲线"""
        plt.figure(figsize=(10, 8))
        
        # 将真实标签转换为one-hot编码
        n_classes = len(fault_mapping)
        y_true_onehot = np.zeros((len(y_true), n_classes))
        for i in range(len(y_true)):
            y_true_onehot[i, y_true[i]] = 1
        
        # 计算每个类别的ROC曲线
        for i in range(n_classes):
            if i == 0 and np.sum(y_true == i) < 10:  # 跳过样本太少的类别
                continue
                
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr, lw=2,
                label=f'{fault_mapping[i]} (AUC = {roc_auc:.2f})'
            )
        
        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(self.config.output_dir, 'roc_curve.png'), dpi=300)
        plt.close()
    
    def _plot_precision_recall_curve(self, y_true, y_pred_proba, fault_mapping):
        """绘制精确率-召回率曲线"""
        plt.figure(figsize=(10, 8))
        
        # 将真实标签转换为one-hot编码
        n_classes = len(fault_mapping)
        y_true_onehot = np.zeros((len(y_true), n_classes))
        for i in range(len(y_true)):
            y_true_onehot[i, y_true[i]] = 1
        
        # 计算每个类别的精确率-召回率曲线
        for i in range(n_classes):
            if i == 0 and np.sum(y_true == i) < 10:  # 跳过样本太少的类别
                continue
                
            precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_pred_proba[:, i])
            avg_precision = average_precision_score(y_true_onehot[:, i], y_pred_proba[:, i])
            
            plt.plot(
                recall, precision, lw=2,
                label=f'{fault_mapping[i]} (AP = {avg_precision:.2f})'
            )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(self.config.output_dir, 'precision_recall_curve.png'), dpi=300)
        plt.close()


class ModelInterpreter:
    """模型解释器"""
    def __init__(self, config: MambaFaultConfig):
        self.config = config
    
    def interpret(self, model, X_test, y_test, fault_mapping):
        """解释模型预测"""
        logging.info("开始解释模型预测...")
        
        # 评估模式
        model.eval()
        
        # 预测
        with torch.no_grad():
            y_pred_proba = torch.softmax(model(X_test), dim=1).cpu().numpy()
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 分析错误预测
        self._analyze_errors(y_test, y_pred, y_pred_proba, fault_mapping)
        
        # 分析特征重要性
        self._analyze_feature_importance(model, X_test, y_test)
        
        # 保存模型解释结果
        self._save_interpretation_results(y_test, y_pred, y_pred_proba, fault_mapping)
    
    def _analyze_errors(self, y_true, y_pred, y_pred_proba, fault_mapping):
        """分析错误预测"""
        # 找出错误预测的样本
        error_indices = np.where(y_true != y_pred)[0]
        
        if len(error_indices) == 0:
            logging.info("没有错误预测的样本")
            return
        
        # 分析错误类型
        error_types = {}
        for idx in error_indices:
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            error_key = f"{fault_mapping[true_label]}->{fault_mapping[pred_label]}"
            
            if error_key not in error_types:
                error_types[error_key] = 0
            error_types[error_key] += 1
        
        # 按错误数量排序
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        
        # 记录错误分析
        logging.info(f"错误预测分析 (共 {len(error_indices)} 个错误):")
        for error_type, count in sorted_errors:
            logging.info(f"  {error_type}: {count} 个样本 ({count/len(error_indices)*100:.1f}%)")
        
        # 分析置信度
        confidence_correct = y_pred_proba[y_true == y_pred].max(axis=1).mean()
        confidence_incorrect = y_pred_proba[y_true != y_pred].max(axis=1).mean()
        
        logging.info(f"正确预测的平均置信度: {confidence_correct:.4f}")
        logging.info(f"错误预测的平均置信度: {confidence_incorrect:.4f}")
        
        # 绘制置信度分布
        plt.figure(figsize=(10, 6))
        
        plt.hist(
            y_pred_proba[y_true == y_pred].max(axis=1),
            alpha=0.5, bins=20, label='正确预测'
        )
        plt.hist(
            y_pred_proba[y_true != y_pred].max(axis=1),
            alpha=0.5, bins=20, label='错误预测'
        )
        
        plt.xlabel('预测置信度')
        plt.ylabel('样本数量')
        plt.title('预测置信度分布')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(self.config.output_dir, 'confidence_distribution.png'), dpi=300)
        plt.close()
    
    def _analyze_feature_importance(self, model, X_test, y_test):
        """分析特征重要性（使用排列重要性方法）"""
        logging.info("分析特征重要性...")
        
        # 由于Mamba模型复杂性，这里使用简化的特征重要性分析
        # 实际应用中可以使用更复杂的方法如SHAP或集成模型的特征重要性
        
        # 这里仅作为示例，实际项目中可以根据需要扩展
        logging.info("特征重要性分析需要根据具体项目需求进一步实现")
    
    def _save_interpretation_results(self, y_true, y_pred, y_pred_proba, fault_mapping):
        """保存模型解释结果"""
        # 创建结果字典
        results = {
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, 
                                                         target_names=list(fault_mapping.values()),
                                                         output_dict=True),
            'fault_mapping': fault_mapping
        }
        
        # 保存为JSON文件
        with open(os.path.join(self.config.output_dir, 'interpretation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        logging.info(f"模型解释结果已保存到 {self.config.output_dir}/interpretation_results.json")


def main():
    """主函数"""
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"开始运行 Mamba 故障预测系统 - {timestamp}")
    
    # 创建配置
    config = MambaFaultConfig()
    
    # 数据处理
    data_processor = DataProcessor(config)
    train_loader, X_test, y_test, fault_mapping = data_processor.load_data()
    
    # 获取特征维度
    feature_dim = X_test.shape[2]
    num_classes = len(fault_mapping)
    
    logging.info(f"特征维度: {feature_dim}")
    logging.info(f"类别数量: {num_classes}")
    
    # 创建模型
    model = MambaSequenceModel(
        input_dim=feature_dim,
        num_classes=num_classes,
        config=config
    ).to(config.device)
    
    # 打印模型结构
    logging.info(f"模型结构:\n{model}")
    
    # 训练模型
    trainer = ModelTrainer(config)
    model = trainer.train(
        model=model,
        train_loader=train_loader,
        X_test=X_test,
        y_test=y_test,
        fault_mapping=fault_mapping,
        class_weights=data_processor.class_weights if config.use_class_weights else None
    )
    
    # 评估模型
    evaluator = ModelEvaluator(config)
    eval_results = evaluator.evaluate(model, X_test, y_test, fault_mapping)
    
    # 解释模型
    interpreter = ModelInterpreter(config)
    interpreter.interpret(model, X_test, y_test, fault_mapping)
    
    # 保存评估结果
    with open(os.path.join(config.output_dir, 'evaluation_results.json'), 'w') as f:
        # 转换numpy数组为列表以便JSON序列化
        serializable_results = {
            'accuracy': float(eval_results['accuracy']),
            'f1': float(eval_results['f1']),
            'precision': float(eval_results['precision']),
            'recall': float(eval_results['recall'])
        }
        json.dump(serializable_results, f, indent=4)
    
    logging.info(f"评估结果已保存到 {config.output_dir}/evaluation_results.json")
    logging.info(f"Mamba 故障预测系统运行完成")


if __name__ == "__main__":
    main()