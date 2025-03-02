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
    f1_score, accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# 导入model.py中的Mamba模型组件
from model import ModelArgs, RMSNorm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mamba_training_reduced.log'),
        logging.StreamHandler()
    ]
)

# 设置 Matplotlib 的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class Config:
    """配置类"""
    data_path: str = "./ai4i2020.csv"
    model_save_path: str = "best_mamba_model_reduced.pth"
    base_output_dir: str = "mamba_experiment_results_reduced"  # 基础输出目录
    experiment_data_path: str = "mamba_evaluation_results_reduced.json"
    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 1e-4
    early_stopping_patience: int = 20  # 早停
    
    # Mamba模型参数
    d_model: int = 128
    n_layer: int = 2
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    
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


class MambaForFaultPrediction(nn.Module):
    """使用Mamba架构的故障预测模型"""
    
    def __init__(self, config: Config, input_dim: int, num_classes: int):
        super().__init__()
        self.config = config
        
        # 创建Mamba模型参数
        self.args = ModelArgs(
            d_model=config.d_model,
            n_layer=config.n_layer,
            vocab_size=num_classes,  # 这里不是真正的词汇表大小，只是为了兼容Mamba的接口
            d_state=config.d_state,
            expand=config.expand,
            dt_rank=config.dt_rank,
            d_conv=config.d_conv
        )
        
        # 特征嵌入层
        self.embedding = nn.Linear(input_dim, config.d_model)
        
        # 位置编码（简化版，因为我们的序列长度为1）
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, config.d_model))
        
        # Mamba层
        from model import ResidualBlock
        self.layers = nn.ModuleList([ResidualBlock(self.args) for _ in range(config.n_layer)])
        
        # 最终归一化
        self.norm_f = RMSNorm(config.d_model)
        
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
        
        # 通过Mamba层
        for layer in self.layers:
            x = layer(x)
        
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
            logits = model(X_test_tensor)
            y_probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = np.argmax(y_probs, axis=1)

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # 打印评估指标
        logging.info(f"准确率: {accuracy:.4f}")
        logging.info(f"宏平均F1分数: {f1_macro:.4f}")
        logging.info(f"加权平均F1分数: {f1_weighted:.4f}")
        
        # 打印分类报告
        class_names = [fault_mapping[i] for i in range(len(fault_mapping))]
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
        
        # 添加是否发生故障的二分类报告
        print("\n是否发生故障分类报告:")
        # 将预测和真实标签转换为二分类（0=无故障，1=有故障）
        y_test_binary = (y_test > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
        binary_class_names = ['无故障', '发生故障']
        print(classification_report(y_test_binary, y_pred_binary, 
                                   target_names=binary_class_names, 
                                   zero_division=0))
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 计算二分类混淆矩阵
        cm_binary = confusion_matrix(y_test_binary, y_pred_binary)
        
        # 保存评估结果数据
        evaluation_results = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            ),
            'binary_classification_report': classification_report(
                y_test_binary, y_pred_binary,
                target_names=binary_class_names,
                output_dict=True,
                zero_division=0
            ),
            'confusion_matrix': cm.tolist(),
            'binary_confusion_matrix': cm_binary.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_probs.tolist()
        }

        # 保存为JSON文件
        json_path = os.path.join(self.config.output_dir, self.config.experiment_data_path)
        with open(json_path, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        logging.info(f"评估结果已保存到 {json_path}")

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
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues',
                   xticklabels=binary_class_names,
                   yticklabels=binary_class_names)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('是否发生故障混淆矩阵')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, "binary_confusion_matrix.svg"), format='svg', bbox_inches='tight')
        plt.close()
        
        # 绘制ROC曲线（多分类）
        plt.figure(figsize=(12, 8))
        
        # 为每个类别计算ROC曲线
        n_classes = len(fault_mapping)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # 将标签转换为one-hot编码
        y_test_bin = np.eye(n_classes)[y_test.astype(int)]
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2,
                     label=f'{fault_mapping[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率')
        plt.ylabel('真正例率')
        plt.title('各类别的ROC曲线')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.config.output_dir, "roc_curves.svg"), format='svg', bbox_inches='tight')
        plt.close()
        
        # 绘制二分类ROC曲线
        plt.figure(figsize=(10, 8))
        # 计算二分类的预测概率（所有故障类别的概率之和）
        y_prob_binary = np.sum(y_probs[:, 1:], axis=1)
        fpr_binary, tpr_binary, _ = roc_curve(y_test_binary, y_prob_binary)
        roc_auc_binary = auc(fpr_binary, tpr_binary)
        
        plt.plot(fpr_binary, tpr_binary, lw=2, label=f'发生故障 vs 无故障 (AUC = {roc_auc_binary:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率')
        plt.ylabel('真正例率')
        plt.title('故障检测ROC曲线')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.config.output_dir, "binary_roc_curve.svg"), format='svg', bbox_inches='tight')
        plt.close()
        
        # 绘制PR曲线（多分类）
        plt.figure(figsize=(12, 8))
        
        # 为每个类别计算PR曲线
        precision = dict()
        recall = dict()
        avg_precision = dict()
        
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_probs[:, i])
            avg_precision[i] = average_precision_score(y_test_bin[:, i], y_probs[:, i])
            plt.plot(recall[i], precision[i], lw=2,
                     label=f'{fault_mapping[i]} (AP = {avg_precision[i]:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('各类别的PR曲线')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(self.config.output_dir, "pr_curves.svg"), format='svg', bbox_inches='tight')
        plt.close()
        
        # 绘制二分类PR曲线
        plt.figure(figsize=(10, 8))
        precision_binary, recall_binary, _ = precision_recall_curve(y_test_binary, y_prob_binary)
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
        
        # 绘制每个类别的精确率和召回率
        plt.figure(figsize=(12, 6))
        report = evaluation_results['classification_report']
        classes = []
        precision_values = []
        recall_values = []
        
        for cls in report:
            if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                try:
                    class_idx = int(cls)
                    class_name = fault_mapping[class_idx]
                except ValueError:
                    # 如果cls不是数字字符串，则它可能已经是类别名称
                    class_name = cls
                
                classes.append(class_name)
                precision_values.append(report[cls]['precision'])
                recall_values.append(report[cls]['recall'])
        
        x = np.arange(len(classes))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, precision_values, width, label='精确率')
        rects2 = ax.bar(x + width/2, recall_values, width, label='召回率')
        
        ax.set_ylabel('分数')
        ax.set_title('各故障类型的精确率和召回率')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, "precision_recall_by_class.svg"), format='svg', bbox_inches='tight')
        plt.close()
        
        return evaluation_results


def plot_learning_curves(train_losses: List[float], val_losses: List[float], output_dir: str):
    """绘制学习曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "learning_curves.svg"), format='svg', bbox_inches='tight')
    plt.close()
    logging.info(f"学习曲线已保存到 {os.path.join(output_dir, 'learning_curves.svg')}")


def analyze_feature_importance(model: nn.Module, feature_names: List[str], output_dir: str):
    """分析特征重要性（基于嵌入层权重）"""
    # 获取嵌入层权重
    embedding_weights = model.embedding.weight.data.cpu().numpy()
    
    # 计算每个特征的重要性（基于权重的绝对值）
    importance = np.mean(np.abs(embedding_weights), axis=0)
    
    # 创建特征重要性数据框
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # 按重要性排序
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # 绘制特征重要性
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('特征重要性分析')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.svg"), format='svg', bbox_inches='tight')
    plt.close()
    
    # 保存特征重要性数据
    feature_importance.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    logging.info(f"特征重要性分析已保存到 {output_dir}")
    
    return feature_importance


def main():
    """主函数"""
    # 初始化配置
    config = Config()
    
    # 数据处理
    data_processor = DataProcessor(config)
    train_loader, test_loader, X_test_tensor, y_test, feature_info = data_processor.load_and_preprocess()
    
    # 获取输入维度
    input_dim = next(iter(train_loader))[0].shape[1]
    
    # 初始化模型
    model = MambaForFaultPrediction(config, input_dim=input_dim, num_classes=config.num_classes).to(config.device)
    logging.info(f"模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 训练模型
    trainer = ModelTrainer(config, model)
    y_test_tensor = torch.LongTensor(y_test).to(config.device)
    train_losses, val_losses = trainer.train(train_loader, X_test_tensor, y_test_tensor)
    
    # 绘制学习曲线
    plot_learning_curves(train_losses, val_losses, config.output_dir)
    
    # 评估模型
    evaluator = ModelEvaluator(config)
    evaluation_results = evaluator.evaluate(model, X_test_tensor, y_test, data_processor.fault_mapping)
    
    # 分析特征重要性
    feature_importance = analyze_feature_importance(model, feature_info['feature_names'], config.output_dir)
    
    # 打印最终结果
    logging.info("模型训练与评估完成！")
    logging.info(f"最佳验证损失: {trainer.best_val_loss:.4f}")
    logging.info(f"准确率: {evaluation_results['accuracy']:.4f}")
    logging.info(f"宏平均F1分数: {evaluation_results['f1_macro']:.4f}")
    logging.info(f"加权平均F1分数: {evaluation_results['f1_weighted']:.4f}")
    
    # 显示最重要的特征
    top_features = feature_importance.head(10)
    logging.info("前10个最重要的特征:")
    for i, (feature, importance) in enumerate(zip(top_features['Feature'], top_features['Importance']), 1):
        logging.info(f"{i}. {feature}: {importance:.4f}")
    
    logging.info(f"所有结果已保存到: {config.output_dir}")
    
    return evaluation_results


if __name__ == "__main__":
    main()