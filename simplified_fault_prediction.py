import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Dict, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simplified_training.log'),
        logging.StreamHandler()
    ]
)

# 设置 Matplotlib 的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义一个简化版的序列模型作为替代
class SimplifiedSequenceModel(nn.Module):
    def __init__(self, d_model, d_state=None, d_conv=None, expand=2, 
                 headdim=None, ngroups=None, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        
        # 简单的自注意力机制
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_inner),
            nn.GELU(),
            nn.Linear(self.d_inner, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        logging.info(f"初始化简化版序列模型，d_model={d_model}")
    
    def forward(self, x):
        # 自注意力层
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output
        
        # 前馈网络
        x = x + self.ffn(self.norm2(x))
        
        return x


@dataclass
class Config:
    """配置类"""
    data_path: str = "./ai4i2020.csv"
    model_save_path: str = "best_simplified_model.pth"
    base_output_dir: str = "simplified_experiment_results"  # 基础输出目录
    experiment_data_path: str = "simplified_evaluation_results.json"
    batch_size: int = 128
    num_epochs: int = 100
    early_stopping_patience: int = 20  # 早停

    # 模型参数
    d_model: int = 192
    n_layer: int = 2
    expand: int = 2
    learning_rate: float = 2e-4
    
    max_seq_len: int = 1  # 表格数据每个样本作为一个序列元素
    device: torch.device = None
    columns_to_drop: List[str] = None
    output_dir: str = None  # 将在 __post_init__ 中设置
    fault_types: List[str] = None  # 故障类型列表
    num_classes: int = 4  # 包括无故障和3种故障类型(HDF, PWF, OSF)

    def __post_init__(self):
        self.columns_to_drop = ['UDI', 'Product ID', 'TWF', 'RNF']  # 删除不相关的ID列和TWF、RNF列
        self.fault_types = ['HDF', 'PWF', 'OSF']  # 只考虑这三种故障类型

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


class SimplifiedFaultPredictionModel(nn.Module):
    """使用简化架构的故障预测模型"""
    
    def __init__(self, config: Config, input_dim: int, num_classes: int):
        super().__init__()
        self.config = config
        
        # 特征嵌入层
        self.embedding = nn.Linear(input_dim, config.d_model)
        
        # 位置编码（简化版，因为我们的序列长度为1）
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, config.d_model))
        
        # 序列处理层
        self.layers = nn.ModuleList([
            SimplifiedSequenceModel(
                d_model=config.d_model,
                expand=config.expand
            ) for _ in range(config.n_layer)
        ])
        
        # 最终归一化
        self.norm_f = nn.LayerNorm(config.d_model)
        
        # 分类头
        self.classifier = nn.Linear(config.d_model, num_classes)
    
    def forward(self, x):
        # x: [batch_size, features]
        
        # 将输入转换为序列形式 [batch_size, 1, features]
        x = x.unsqueeze(1)
        
        # 特征嵌入 [batch_size, 1, d_model]
        x = self.embedding(x)
        
        # 添加位置编码
        x = x + self.pos_encoder
        
        # 通过序列处理层
        for layer in self.layers:
            x = x + layer(x)  # 添加残差连接
        
        # 最终归一化
        x = self.norm_f(x)
        
        # 取序列的第一个元素作为分类输入
        x = x.squeeze(1)
        
        # 分类
        return self.classifier(x)


class DataProcessor:
    """数据处理类"""

    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()  # 用于编码产品类型
        self.fault_mapping = None  # 故障类型映射

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
        
        # 添加温差特征
        df['Temp_diff'] = df['Air temperature [K]'] - df['Process temperature [K]']
        logging.info("添加温差特征")
        
        # 编码产品类型
        df['Type_encoded'] = self.label_encoder.fit_transform(df['Type'])
        logging.info(f"产品类型编码映射: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # 创建故障类型标签
        # 首先检查每行是否有故障
        df['Has_Failure'] = df['Machine failure'].astype(bool)
        
        # 创建故障类型标签（0表示无故障，1-3表示不同类型的故障）
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
        
        # 数据集划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 转换为张量并移动到指定设备
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.config.device)
        y_train_tensor = torch.LongTensor(y_train.values).to(self.config.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.config.device)
        y_test_tensor = torch.LongTensor(y_test.values).to(self.config.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
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
        
        # 保存特征重要性信息
        feature_info = {
            'feature_names': X.columns.tolist(),
            'fault_mapping': self.fault_mapping,
            'type_mapping': dict(zip(range(len(self.label_encoder.classes_)), self.label_encoder.classes_))
        }
        
        return train_loader, test_loader, X_test_tensor, y_test.values, feature_info


class ModelTrainer:
    """模型训练类"""

    def __init__(self, config: Config, model: nn.Module):
        self.config = config
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, X_test_tensor: torch.Tensor, y_test_tensor: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            val_loss = self.criterion(outputs, y_test_tensor)
        return val_loss.item()

    def train(self, train_loader: DataLoader, X_test_tensor: torch.Tensor,
              y_test_tensor: torch.Tensor) -> Tuple[List[float], List[float]]:
        train_losses, val_losses = [], []

        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(X_test_tensor, y_test_tensor)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            logging.info(f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                         f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict()
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        # 保存最佳模型
        if self.best_model_state is not None:
            torch.save(self.best_model_state, self.config.model_save_path)
            logging.info(f"最佳模型已保存到 {self.config.model_save_path}")
            self.model.load_state_dict(self.best_model_state)

        return train_losses, val_losses


class ModelEvaluator:
    """模型评估类"""

    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, model: nn.Module, X_test_tensor: torch.Tensor, y_test: np.ndarray, fault_mapping: Dict[int, str]):
        """评估模型性能并生成可视化结果"""
        # 确保数据在正确的设备上
        X_test_tensor = X_test_tensor.to(self.config.device, non_blocking=True)
        
        model.eval()
        with torch.no_grad():
            # 获取模型预测
            outputs = model(X_test_tensor)
            _, y_pred = torch.max(outputs, 1)
            y_pred = y_pred.cpu().numpy()
            
            # 获取概率分布
            y_probs = torch.softmax(outputs, dim=1).cpu().numpy()
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        logging.info(f"准确率: {accuracy:.4f}")
        logging.info(f"F1分数(宏平均): {f1_macro:.4f}")
        logging.info(f"F1分数(加权平均): {f1_weighted:.4f}")
        
        # 生成分类报告
        class_names = [fault_mapping[i] for i in range(len(fault_mapping))]
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        logging.info(f"分类报告:\n{classification_report(y_test, y_pred, target_names=class_names)}")
        
        # 生成混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 创建二分类标签（有故障/无故障）
        y_test_binary = (y_test > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
        
        # 计算二分类的概率（所有故障类型的概率之和）
        y_prob_binary = np.sum(y_probs[:, 1:], axis=1)
        
        # 二分类评估指标
        binary_accuracy = accuracy_score(y_test_binary, y_pred_binary)
        binary_f1 = f1_score(y_test_binary, y_pred_binary)
        binary_precision = precision_score(y_test_binary, y_pred_binary)
        binary_recall = recall_score(y_test_binary, y_pred_binary)
        
        logging.info(f"二分类准确率: {binary_accuracy:.4f}")
        logging.info(f"二分类F1分数: {binary_f1:.4f}")
        logging.info(f"二分类精确率: {binary_precision:.4f}")
        logging.info(f"二分类召回率: {binary_recall:.4f}")
        
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
            }
        }
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
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
        plt.figure(figsize=(10, 8))
        
        # 多分类ROC曲线
        for i in range(len(fault_mapping)):
            if i == 0:  # 无故障类别
                label = f'{fault_mapping[i]} vs 其他'
            else:
                label = f'{fault_mapping[i]} vs 其他'
            
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
        
        # 二分类ROC曲线
        fpr_binary, tpr_binary, _ = roc_curve(y_test_binary, y_prob_binary)
        roc_auc_binary = auc(fpr_binary, tpr_binary)
        plt.plot(fpr_binary, tpr_binary, lw=3, label=f'有故障 vs 无故障 (AUC = {roc_auc_binary:.2f})', linestyle='--')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率')
        plt.ylabel('真正例率')
        plt.title('ROC曲线')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.config.output_dir, "roc_curve.svg"), format='svg', bbox_inches='tight')
        plt.close()
        
        # 绘制二分类PR曲线
        plt.figure(figsize=(10, 8))
        precision_binary, recall_binary, thresholds_pr = precision_recall_curve(y_test_binary, y_prob_binary)
        avg_precision_binary = average_precision_score(y_test_binary, y_prob_binary)
        
        plt.plot(recall_binary, precision_binary, lw=2, label=f'发生故障 vs 无故障 (AP = {avg_precision_binary:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('故障检测PR曲线')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(self.config.output_dir, "binary_pr_curve.svg"), format='svg', bbox_inches='tight')
        plt.close()
        
        # 计算最佳阈值（使F1分数最大化）
        f1_scores = []
        for i in range(len(thresholds_pr)):
            # 使用当前阈值进行预测
            y_pred_temp = (y_prob_binary >= thresholds_pr[i]).astype(int)
            # 计算F1分数
            f1 = f1_score(y_test_binary, y_pred_temp)
            f1_scores.append(f1)
        
        # 找到最大F1分数对应的索引
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds_pr[best_idx]
        best_f1 = f1_scores[best_idx]
        
        # 使用最佳阈值重新预测
        y_pred_binary_optimal = (y_prob_binary >= best_threshold).astype(int)
        
        # 计算最佳阈值下的性能指标
        optimal_precision = precision_score(y_test_binary, y_pred_binary_optimal)
        optimal_recall = recall_score(y_test_binary, y_pred_binary_optimal)
        optimal_f1 = f1_score(y_test_binary, y_pred_binary_optimal)
        
        # 记录最佳阈值信息
        logging.info(f"最佳阈值: {best_threshold:.4f}")
        logging.info(f"最佳阈值下的F1分数: {optimal_f1:.4f}")
        logging.info(f"最佳阈值下的精确率: {optimal_precision:.4f}")
        logging.info(f"最佳阈值下的召回率: {optimal_recall:.4f}")
        
        # 绘制F1分数与阈值的关系图
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds_pr, f1_scores, 'b-')
        plt.axvline(x=best_threshold, color='r', linestyle='--', 
                   label=f'最佳阈值 = {best_threshold:.4f}\nF1 = {best_f1:.4f}')
        plt.xlabel('阈值')
        plt.ylabel('F1分数')
        plt.title('F1分数与阈值的关系')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config.output_dir, "f1_threshold.svg"), format='svg', bbox_inches='tight')
        plt.close()
        
        # 将最佳阈值信息添加到评估结果中
        evaluation_results['optimal_threshold'] = float(best_threshold)
        evaluation_results['optimal_binary_metrics'] = {
            'precision': float(optimal_precision),
            'recall': float(optimal_recall),
            'f1': float(optimal_f1),
            'classification_report': classification_report(
                y_test_binary, y_pred_binary_optimal,
                target_names=binary_class_names,
                output_dict=True,
                zero_division=0
            ),
            'confusion_matrix': confusion_matrix(y_test_binary, y_pred_binary_optimal).tolist()
        }
        
        # 绘制最佳阈值下的混淆矩阵
        plt.figure(figsize=(8, 6))
        cm_binary_optimal = confusion_matrix(y_test_binary, y_pred_binary_optimal)
        sns.heatmap(cm_binary_optimal, annot=True, fmt='d', cmap='Blues',
                   xticklabels=binary_class_names,
                   yticklabels=binary_class_names)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'最佳阈值({best_threshold:.4f})下的二分类混淆矩阵')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, "optimal_binary_confusion_matrix.svg"), format='svg', bbox_inches='tight')
        plt.close()
        
        # 保存评估结果
        with open(os.path.join(self.config.output_dir, self.config.experiment_data_path), 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        
        logging.info(f"评估结果已保存到 {os.path.join(self.config.output_dir, self.config.experiment_data_path)}")
        
        return evaluation_results


def plot_training_history(train_losses, val_losses, output_dir):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练历史')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_history.svg"), format='svg', bbox_inches='tight')
    plt.close()


def main():
    """主函数"""
    # 初始化配置
    config = Config()
    logging.info("配置初始化完成")
    
    # 数据处理
    data_processor = DataProcessor(config)
    train_loader, test_loader, X_test_tensor, y_test, feature_info = data_processor.load_and_preprocess()
    logging.info("数据处理完成")
    
    # 获取输入维度
    input_dim = X_test_tensor.shape[1]
    
    # 创建模型
    model = SimplifiedFaultPredictionModel(config, input_dim, config.num_classes)
    model = model.to(config.device)
    logging.info(f"模型创建完成，参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 训练模型
    trainer = ModelTrainer(config, model)
    train_losses, val_losses = trainer.train(train_loader, X_test_tensor, torch.LongTensor(y_test).to(config.device))
    logging.info("模型训练完成")
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, config.output_dir)
    
    # 评估模型
    evaluator = ModelEvaluator(config)
    evaluation_results = evaluator.evaluate(model, X_test_tensor, y_test, data_processor.fault_mapping)
    logging.info("模型评估完成")
    
    
    # 打印最终结果
    logging.info(f"实验完成！结果保存在: {config.output_dir}")
    logging.info(f"准确率: {evaluation_results['accuracy']:.4f}")
    logging.info(f"F1分数(加权平均): {evaluation_results['f1_weighted']:.4f}")
    logging.info(f"最佳阈值: {evaluation_results['optimal_threshold']:.4f}")
    logging.info(f"最佳阈值下的F1分数: {evaluation_results['optimal_binary_metrics']['f1']:.4f}")
    
    return model, evaluation_results


if __name__ == "__main__":
    main()