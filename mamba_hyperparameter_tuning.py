import json
import logging
import os
import optuna
from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, List, Dict, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# 导入原始模型组件
from model import ModelArgs, RMSNorm, ResidualBlock
from mamba_fault_prediction_reduced import MambaForFaultPrediction, DataProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mamba_hyperparameter_tuning.log'),
        logging.StreamHandler()
    ]
)

# 设置 Matplotlib 的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class TuningConfig:
    """超参数调优配置类"""
    data_path: str = "./ai4i2020.csv"
    base_output_dir: str = "mamba_hyperparameter_tuning"
    n_trials: int = 30  # Optuna试验次数
    cv_folds: int = 3   # 交叉验证折数
    
    # 固定参数
    batch_size: int = 128
    num_epochs: int = 50  # 每次试验的训练轮数减少，加快调优速度
    early_stopping_patience: int = 10
    max_seq_len: int = 1
    
    # 超参数搜索空间
    d_model_range: Tuple[int, int] = (32, 256)
    n_layer_range: Tuple[int, int] = (1, 4)
    d_state_range: Tuple[int, int] = (8, 64)
    expand_range: Tuple[int, int] = (1, 4)
    d_conv_range: Tuple[int, int] = (2, 8)
    learning_rate_range: Tuple[float, float] = (1e-5, 1e-3)
    
    # 运行时设置
    device: torch.device = None
    columns_to_drop: List[str] = field(default_factory=list)
    output_dir: str = None
    fault_types: List[str] = field(default_factory=list)
    num_classes: int = 4

    def __post_init__(self):
        self.columns_to_drop = ['UDI', 'Product ID', 'TWF', 'RNF']
        self.fault_types = ['HDF', 'PWF', 'OSF']

        # 创建基础输出目录
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)

        # 使用当前时间创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.base_output_dir, timestamp)

        # 创建实验目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 检测并使用最快的可用设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logging.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logging.info("使用 CPU")


class HyperparameterTuner:
    """超参数调优类"""
    
    def __init__(self, config: TuningConfig):
        self.config = config
        self.best_params = None
        self.best_score = 0.0
        self.best_model = None
        self.results = []
        self.study = None
        
        # 数据处理
        self.data_processor = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.feature_names = None
        self.fault_mapping = None
        
    def prepare_data(self):
        """准备数据集"""
        logging.info("准备数据集...")
        
        # 创建一个临时配置对象用于数据处理
        from mamba_fault_prediction_reduced import Config
        temp_config = Config()
        temp_config.data_path = self.config.data_path
        temp_config.columns_to_drop = self.config.columns_to_drop
        temp_config.fault_types = self.config.fault_types
        temp_config.device = self.config.device
        
        # 使用原始数据处理器加载数据
        self.data_processor = DataProcessor(temp_config)
        
        # 加载数据
        try:
            df = pd.read_csv(self.config.data_path)
            logging.info("数据加载成功！")
        except FileNotFoundError:
            logging.error(f"错误：文件未找到，请检查路径 {self.config.data_path}")
            raise
            
        # 数据预处理（与原始代码类似）
        df['Temp_diff'] = df['Air temperature [K]'] - df['Process temperature [K]']
        df['Type_encoded'] = self.data_processor.label_encoder.fit_transform(df['Type'])
        
        # 创建故障类型标签
        df['Has_Failure'] = df['Machine failure'].astype(bool)
        df['Fault_Type'] = 0  # 默认为无故障
        
        for i, fault_type in enumerate(self.config.fault_types, 1):
            mask = df[fault_type] == 1
            df.loc[mask, 'Fault_Type'] = i
        
        # 创建故障类型映射
        self.data_processor.fault_mapping = {0: '无故障'}
        for i, fault_type in enumerate(self.config.fault_types, 1):
            self.data_processor.fault_mapping[i] = fault_type
        
        self.fault_mapping = self.data_processor.fault_mapping
        
        # 数据预处理
        columns_to_drop = self.config.columns_to_drop + ['Machine failure', 'Type', 'Has_Failure'] + self.config.fault_types
        df_processed = df.drop(columns=columns_to_drop, errors='ignore')
        
        # 分离特征和标签
        X = df_processed.drop(columns=['Fault_Type'])
        y = df['Fault_Type']
        self.feature_names = X.columns.tolist()
        
        # 数据集划分
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 标准化
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_val_scaled = scaler.transform(self.X_val)
        
        logging.info(f"数据准备完成。训练集: {self.X_train.shape}, 验证集: {self.X_val.shape}")
        
        return self.X_train.shape[1]  # 返回特征维度
    
    def objective(self, trial):
        """Optuna优化目标函数"""
        # 从试验中采样超参数
        params = {
            'd_model': trial.suggest_int('d_model', self.config.d_model_range[0], self.config.d_model_range[1]),
            'n_layer': trial.suggest_int('n_layer', self.config.n_layer_range[0], self.config.n_layer_range[1]),
            'd_state': trial.suggest_int('d_state', self.config.d_state_range[0], self.config.d_state_range[1]),
            'expand': trial.suggest_int('expand', self.config.expand_range[0], self.config.expand_range[1]),
            'd_conv': trial.suggest_int('d_conv', self.config.d_conv_range[0], self.config.d_conv_range[1]),
            'learning_rate': trial.suggest_float('learning_rate', 
                                               self.config.learning_rate_range[0], 
                                               self.config.learning_rate_range[1], 
                                               log=True)
        }
        
        # 记录当前试验的超参数
        logging.info(f"试验 #{trial.number} 超参数: {params}")
        
        # 创建模型
        model = self.create_model(params)
        
        # 训练模型
        val_metrics = self.train_and_evaluate(model, params)
        
        # 记录结果
        self.results.append({
            'trial': trial.number,
            'params': params,
            'metrics': val_metrics
        })
        
        # 使用F1加权平均分数作为优化目标
        return val_metrics['f1_weighted']
    
    def create_model(self, params):
        """创建模型"""
        # 创建一个临时配置对象
        from mamba_fault_prediction_reduced import Config
        model_config = Config()
        model_config.d_model = params['d_model']
        model_config.n_layer = params['n_layer']
        model_config.d_state = params['d_state']
        model_config.expand = params['expand']
        model_config.d_conv = params['d_conv']
        model_config.device = self.config.device
        model_config.num_classes = self.config.num_classes
        
        # 创建模型
        input_dim = self.X_train.shape[1]
        model = MambaForFaultPrediction(model_config, input_dim=input_dim, num_classes=self.config.num_classes)
        model = model.to(self.config.device)
        
        return model
    
    def train_and_evaluate(self, model, params):
        """训练和评估模型"""
        # 准备数据
        X_train_tensor = torch.FloatTensor(self.X_train_scaled).to(self.config.device)
        y_train_tensor = torch.LongTensor(self.y_train.values).to(self.config.device)
        X_val_tensor = torch.FloatTensor(self.X_val_scaled).to(self.config.device)
        y_val_tensor = torch.LongTensor(self.y_val.values).to(self.config.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True
        )
        
        # 设置训练参数
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                
                # 计算预测结果
                val_probs = torch.softmax(val_outputs, dim=1).cpu().numpy()
                val_preds = np.argmax(val_probs, axis=1)
                
                # 计算评估指标
                accuracy = accuracy_score(self.y_val, val_preds)
                f1_macro = f1_score(self.y_val, val_preds, average='macro')
                f1_weighted = f1_score(self.y_val, val_preds, average='weighted')
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_metrics = {
                    'accuracy': float(accuracy),
                    'f1_macro': float(f1_macro),
                    'f1_weighted': float(f1_weighted),
                    'val_loss': float(val_loss)
                }
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logging.info(f"早停触发，在第 {epoch + 1} 轮停止训练")
                    break
            
            # 记录训练进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logging.info(f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                            f"Accuracy: {accuracy:.4f} | F1 (weighted): {f1_weighted:.4f}")
        
        # 保存最佳模型状态
        if f1_weighted > self.best_score:
            self.best_score = f1_weighted
            self.best_params = params
            self.best_model = model
            self.best_model.load_state_dict(best_model_state)
        
        return best_metrics
    
    def run_optimization(self):
        """运行超参数优化"""
        logging.info("开始超参数优化...")
        
        # 准备数据
        input_dim = self.prepare_data()
        
        # 创建Optuna研究
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self.objective, n_trials=self.config.n_trials)
        
        # 获取最佳参数
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        # 记录最佳结果
        logging.info(f"超参数优化完成！")
        logging.info(f"最佳参数: {self.best_params}")
        logging.info(f"最佳F1加权分数: {self.best_score:.4f}")
        
        # 保存结果
        self.save_results()
        
        return self.best_params, self.best_score
    
    def save_results(self):
        """保存优化结果"""
        # 保存最佳模型
        if self.best_model is not None:
            model_path = os.path.join(self.config.output_dir, "best_model.pth")
            torch.save(self.best_model.state_dict(), model_path)
            logging.info(f"最佳模型已保存到 {model_path}")
        
        # 保存优化结果
        results_path = os.path.join(self.config.output_dir, "optimization_results.json")
        
        # 将结果转换为可序列化的格式
        serializable_results = []
        for result in self.results:
            metrics = result['metrics'].copy()
            serializable_results.append({
                'trial': result['trial'],
                'params': result['params'],
                'metrics': metrics
            })
        
        # 添加最佳参数和分数
        optimization_results = {
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'all_trials': serializable_results
        }
        
        with open(results_path, 'w') as f:
            json.dump(optimization_results, f, indent=4)
        
        logging.info(f"优化结果已保存到 {results_path}")
        
        # 绘制参数重要性图
        self.plot_parameter_importance()
        
        # 绘制优化历史
        self.plot_optimization_history()
        
        # 绘制参数关系图
        self.plot_param_relationships()
    
    def plot_parameter_importance(self):
        """绘制参数重要性图"""
        if self.study is None:
            return
        
        try:
            # 计算参数重要性
            importances = optuna.importance.get_param_importances(self.study)
            
            # 创建数据框
            importance_df = pd.DataFrame(
                {'Parameter': list(importances.keys()), 
                 'Importance': list(importances.values())}
            )
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # 绘制参数重要性图
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Parameter', data=importance_df)
            plt.title('参数重要性')
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.output_dir, "parameter_importance.svg"), format='svg', bbox_inches='tight')
            plt.close()
            
            logging.info(f"参数重要性图已保存")
        except Exception as e:
            logging.warning(f"绘制参数重要性图失败: {e}")
    
    def plot_optimization_history(self):
        """绘制优化历史"""
        if self.study is None:
            return
        
        # 绘制优化历史
        plt.figure(figsize=(10, 6))
        
        # 提取每次试验的目标值
        trials = self.study.trials
        values = [t.value for t in trials if t.value is not None]
        best_values = np.maximum.accumulate(values)
        
        # 绘制每次试验的目标值和最佳目标值
        plt.plot(values, 'o-', alpha=0.6, label='每次试验的F1分数')
        plt.plot(best_values, 'o-', label='最佳F1分数')
        
        plt.xlabel('试验次数')
        plt.ylabel('F1加权分数')
        plt.title('超参数优化历史')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, "optimization_history.svg"), format='svg', bbox_inches='tight')
        plt.close()
        
        logging.info(f"优化历史图已保存")
    
    def plot_param_relationships(self):
        """绘制参数关系图"""
        if self.study is None or len(self.study.trials) < 5:
            return
        
        try:
            # 提取参数和目标值
            trials_df = self.study.trials_dataframe()
            
            # 选择最重要的参数（如果有参数重要性信息）
            try:
                importances = optuna.importance.get_param_importances(self.study)
                top_params = list(importances.keys())[:3]  # 取前3个最重要的参数
            except:
                # 如果无法计算参数重要性，则使用所有参数
                top_params = [p for p in trials_df.columns if p.startswith('params_')]
            
            # 绘制参数对图
            if len(top_params) >= 2:
                plt.figure(figsize=(12, 10))
                sns.pairplot(
                    trials_df, 
                    vars=top_params, 
                    hue='value',  # 使用目标值作为颜色
                    palette='viridis',
                    diag_kind='kde',
                    plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                    diag_kws={'shade': True}
                )
                plt.suptitle('参数关系图', y=1.02)
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.output_dir, "param_relationships.svg"), format='svg', bbox_inches='tight')
                plt.close()
                
                logging.info(f"参数关系图已保存")
        except Exception as e:
            logging.warning(f"绘制参数关系图失败: {e}")
    
    def apply_best_params_to_model(self):
        """使用最佳参数创建并训练完整模型"""
        if self.best_params is None:
            logging.warning("没有找到最佳参数，无法创建模型")
            return None
        
        logging.info("使用最佳参数创建并训练完整模型...")
        
        # 创建模型
        model = self.create_model(self.best_params)
        
        # 准备数据
        X_train_tensor = torch.FloatTensor(self.X_train_scaled).to(self.config.device)
        y_train_tensor = torch.LongTensor(self.y_train.values).to(self.config.device)
        X_val_tensor = torch.FloatTensor(self.X_val_scaled).to(self.config.device)
        y_val_tensor = torch.LongTensor(self.y_val.values).to(self.config.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True
        )
        
        # 设置训练参数
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.best_params['learning_rate'])
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # 增加训练轮数，确保模型充分训练
        num_epochs = self.config.num_epochs * 2
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                
                # 计算预测结果
                val_probs = torch.softmax(val_outputs, dim=1).cpu().numpy()
                val_preds = np.argmax(val_probs, axis=1)
                
                # 计算评估指标
                accuracy = accuracy_score(self.y_val, val_preds)
                f1_weighted = f1_score(self.y_val, val_preds, average='weighted')
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logging.info(f"早停触发，在第 {epoch + 1} 轮停止训练")
                    break
            
            # 记录训练进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logging.info(f"最终模型训练 - Epoch {epoch + 1}/{num_epochs} | "
                            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                            f"Accuracy: {accuracy:.4f} | F1 (weighted): {f1_weighted:.4f}")
        
        # 加载最佳模型状态
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        # 保存最终模型
        final_model_path = os.path.join(self.config.output_dir, "final_best_model.pth")
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"最终模型已保存到 {final_model_path}")
        
        return model


def main():
    """主函数"""
    # 初始化配置
    config = TuningConfig()
    
    # 创建超参数调优器
    tuner = HyperparameterTuner(config)
    
    # 运行超参数优化
    best_params, best_score = tuner.run_optimization()
    
    # 使用最佳参数训练最终模型
    final_model = tuner.apply_best_params_to_model()
    
    # 打印最佳参数和分数
    logging.info("超参数优化完成！")
    logging.info(f"最佳参数: {best_params}")
    logging.info(f"最佳F1加权分数: {best_score:.4f}")
    
    return best_params, best_score, final_model


if __name__ == "__main__":
    main()