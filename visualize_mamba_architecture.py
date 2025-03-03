import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_mamba_architecture_diagram(output_dir="./mamba_visualizations", filename="mamba_architecture.svg"):
    """
    创建Mamba神经网络架构示意图并保存为SVG文件
    
    Args:
        output_dir: 输出目录
        filename: 输出文件名
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 设置背景色为浅灰色
    ax.set_facecolor('#f5f5f5')
    
    # 定义颜色
    colors = {
        'input': '#3498db',       # 蓝色
        'embedding': '#2ecc71',   # 绿色
        'mamba_block': '#e74c3c', # 红色
        'norm': '#9b59b6',        # 紫色
        'classifier': '#f39c12',  # 橙色
        'output': '#1abc9c',      # 青绿色
        'arrow': '#34495e',       # 深灰色
        'text': '#2c3e50'         # 深蓝灰色
    }
    
    # 定义组件位置和大小
    components = [
        {'name': '输入特征', 'type': 'input', 'pos': (2, 9), 'width': 3, 'height': 1},
        {'name': '特征嵌入层', 'type': 'embedding', 'pos': (2, 7.5), 'width': 3, 'height': 1},
        {'name': '位置编码', 'type': 'embedding', 'pos': (6, 7.5), 'width': 2, 'height': 1},
        {'name': 'Mamba Block 1', 'type': 'mamba_block', 'pos': (2, 6), 'width': 3, 'height': 1},
        {'name': 'Mamba Block 2', 'type': 'mamba_block', 'pos': (2, 4.5), 'width': 3, 'height': 1},
        {'name': '...', 'type': 'text', 'pos': (3.5, 3.5), 'width': 0, 'height': 0},
        {'name': 'Mamba Block N', 'type': 'mamba_block', 'pos': (2, 2.5), 'width': 3, 'height': 1},
        {'name': '最终归一化', 'type': 'norm', 'pos': (2, 1), 'width': 3, 'height': 0.8},
        {'name': '分类头', 'type': 'classifier', 'pos': (7, 1), 'width': 3, 'height': 0.8},
        {'name': '输出预测', 'type': 'output', 'pos': (12, 1), 'width': 2.5, 'height': 0.8}
    ]
    
    # 绘制组件
    for comp in components:
        if comp['type'] == 'text':
            ax.text(comp['pos'][0], comp['pos'][1], comp['name'], 
                   ha='center', va='center', fontsize=14, color=colors['text'], fontweight='bold')
        else:
            rect = patches.Rectangle(
                (comp['pos'][0], comp['pos'][1]), 
                comp['width'], comp['height'],
                linewidth=2,
                edgecolor=colors[comp['type']],
                facecolor=colors[comp['type']] + '50',  # 添加透明度
                alpha=0.9,
                zorder=2
            )
            ax.add_patch(rect)
            ax.text(comp['pos'][0] + comp['width']/2, comp['pos'][1] + comp['height']/2, 
                   comp['name'], ha='center', va='center', fontsize=12, color=colors['text'], fontweight='bold')
    
    # 绘制连接箭头
    connections = [
        (components[0], components[1]),  # 输入特征 -> 特征嵌入层
        (components[1], components[3]),  # 特征嵌入层 -> Mamba Block 1
        (components[2], components[3], 'sum'),  # 位置编码 -> Mamba Block 1 (加法)
        (components[3], components[4]),  # Mamba Block 1 -> Mamba Block 2
        (components[4], components[6], 'skip'),  # Mamba Block 2 -> Mamba Block N (跳过中间块)
        (components[6], components[7]),  # Mamba Block N -> 最终归一化
        (components[7], components[8]),  # 最终归一化 -> 分类头
        (components[8], components[9])   # 分类头 -> 输出预测
    ]
    
    for i, conn in enumerate(connections):
        if len(conn) == 2:
            start, end = conn
            conn_type = 'normal'
        else:
            start, end, conn_type = conn
        
        if conn_type == 'normal':
            # 普通连接
            if start['pos'][1] > end['pos'][1]:  # 垂直连接
                start_x = start['pos'][0] + start['width'] / 2
                start_y = start['pos'][1]
                end_x = end['pos'][0] + end['width'] / 2
                end_y = end['pos'][1] + end['height']
                
                ax.arrow(start_x, start_y, 0, -(start_y - end_y), 
                        head_width=0.2, head_length=0.2, fc=colors['arrow'], ec=colors['arrow'],
                        length_includes_head=True, linewidth=2, zorder=1)
            else:  # 水平连接
                start_x = start['pos'][0] + start['width']
                start_y = start['pos'][1] + start['height'] / 2
                end_x = end['pos'][0]
                end_y = end['pos'][1] + end['height'] / 2
                
                ax.arrow(start_x, start_y, end_x - start_x, 0, 
                        head_width=0.2, head_length=0.2, fc=colors['arrow'], ec=colors['arrow'],
                        length_includes_head=True, linewidth=2, zorder=1)
        
        elif conn_type == 'sum':
            # 加法连接
            start_x = start['pos'][0] + start['width'] / 2
            start_y = start['pos'][1]
            end_x = end['pos'][0] + end['width'] / 2
            end_y = end['pos'][1] + end['height'] / 2
            
            # 绘制曲线路径
            verts = [
                (start_x, start_y),  # 起点
                (start_x, end_y + 0.5),  # 控制点1
                (end_x - 0.5, end_y),  # 控制点2
                (end_x, end_y),  # 终点
            ]
            codes = [
                Path.MOVETO,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
            ]
            
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', edgecolor=colors['arrow'], linewidth=2, zorder=1)
            ax.add_patch(patch)
            
            # 添加加号
            ax.text(end_x - 0.3, end_y + 0.2, '+', fontsize=16, color=colors['arrow'], fontweight='bold')
            
        elif conn_type == 'skip':
            # 跳过连接（虚线）
            start_x = start['pos'][0] + start['width'] / 2
            start_y = start['pos'][1]
            end_x = end['pos'][0] + end['width'] / 2
            end_y = end['pos'][1] + end['height']
            
            ax.plot([start_x, end_x], [start_y, end_y], '--', color=colors['arrow'], linewidth=2, zorder=1)
    
    # 添加Mamba Block的详细结构
    detail_x, detail_y = 8, 5
    detail_width, detail_height = 5, 3
    
    # 绘制详细结构的背景
    detail_rect = patches.Rectangle(
        (detail_x, detail_y), 
        detail_width, detail_height,
        linewidth=2,
        edgecolor='#7f8c8d',
        facecolor='#ecf0f1',
        alpha=0.9,
        zorder=3
    )
    ax.add_patch(detail_rect)
    
    # 添加标题
    ax.text(detail_x + detail_width/2, detail_y + detail_height - 0.3, 
           'Mamba Block 内部结构', ha='center', va='center', 
           fontsize=14, color=colors['text'], fontweight='bold')
    
    # 添加内部组件
    inner_components = [
        {'name': 'RMSNorm', 'pos': (detail_x + 0.5, detail_y + 1.8), 'width': 1.5, 'height': 0.6},
        {'name': 'SSM', 'pos': (detail_x + 2.5, detail_y + 1.8), 'width': 1.5, 'height': 0.6},
        {'name': 'Linear', 'pos': (detail_x + 2.5, detail_y + 0.8), 'width': 1.5, 'height': 0.6}
    ]
    
    for comp in inner_components:
        rect = patches.Rectangle(
            (comp['pos'][0], comp['pos'][1]), 
            comp['width'], comp['height'],
            linewidth=1.5,
            edgecolor='#7f8c8d',
            facecolor='#bdc3c7',
            alpha=0.9,
            zorder=4
        )
        ax.add_patch(rect)
        ax.text(comp['pos'][0] + comp['width']/2, comp['pos'][1] + comp['height']/2, 
               comp['name'], ha='center', va='center', fontsize=10, color=colors['text'])
    
    # 添加内部连接
    # 输入到RMSNorm
    ax.arrow(detail_x, detail_y + 2.1, 0.5, 0, 
            head_width=0.1, head_length=0.1, fc='#7f8c8d', ec='#7f8c8d',
            length_includes_head=True, linewidth=1.5, zorder=3)
    
    # RMSNorm到SSM
    ax.arrow(inner_components[0]['pos'][0] + inner_components[0]['width'], 
            inner_components[0]['pos'][1] + inner_components[0]['height']/2,
            inner_components[1]['pos'][0] - (inner_components[0]['pos'][0] + inner_components[0]['width']), 0,
            head_width=0.1, head_length=0.1, fc='#7f8c8d', ec='#7f8c8d',
            length_includes_head=True, linewidth=1.5, zorder=3)
    
    # SSM到Linear
    ax.arrow(inner_components[1]['pos'][0] + inner_components[1]['width']/2, 
            inner_components[1]['pos'][1],
            0, -(inner_components[1]['pos'][1] - (inner_components[2]['pos'][1] + inner_components[2]['height'])),
            head_width=0.1, head_length=0.1, fc='#7f8c8d', ec='#7f8c8d',
            length_includes_head=True, linewidth=1.5, zorder=3)
    
    # Linear到输出
    ax.arrow(inner_components[2]['pos'][0] + inner_components[2]['width'], 
            inner_components[2]['pos'][1] + inner_components[2]['height']/2,
            detail_x + detail_width - (inner_components[2]['pos'][0] + inner_components[2]['width']), 0,
            head_width=0.1, head_length=0.1, fc='#7f8c8d', ec='#7f8c8d',
            length_includes_head=True, linewidth=1.5, zorder=3)
    
    # 残差连接
    ax.plot([detail_x, detail_x + detail_width], [detail_y + 2.5, detail_y + 2.5], 
           '-', color='#7f8c8d', linewidth=1.5, zorder=3)
    ax.text(detail_x + detail_width/2, detail_y + 2.7, '残差连接', ha='center', fontsize=10, color='#7f8c8d')
    
    # 添加SSM细节说明
    ax.text(detail_x + 4.5, detail_y + 1.8, 'S4/S6 核心', ha='center', va='center', 
           fontsize=8, color='#7f8c8d', style='italic')
    
    # 设置坐标轴
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 添加标题
    plt.suptitle('Mamba 神经网络架构示意图', fontsize=20, y=0.98)
    
    # 添加说明文字
    description = (
        "Mamba是一种高效的序列建模架构，结合了SSM (State Space Model) 和Transformer的优点。\n"
        "它通过可选择性扫描机制实现了线性计算复杂度，同时保持了长序列建模能力。\n"
        "核心组件是S4/S6状态空间模型，可以高效处理长距离依赖关系。"
    )
    plt.figtext(0.5, 0.02, description, ha='center', fontsize=12, 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.7))
    
    # 保存图形
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close()
    
    logging.info(f"Mamba架构示意图已保存到: {output_path}")
    
    return output_path


def create_ssm_detail_diagram(output_dir="./mamba_visualizations", filename="ssm_detail.svg"):
    """
    创建SSM(状态空间模型)详细结构示意图并保存为SVG文件
    
    Args:
        output_dir: 输出目录
        filename: 输出文件名
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置背景色为浅灰色
    ax.set_facecolor('#f5f5f5')
    
    # 定义颜色
    colors = {
        'input': '#3498db',       # 蓝色
        'state': '#e74c3c',       # 红色
        'output': '#2ecc71',      # 绿色
        'param': '#9b59b6',       # 紫色
        'arrow': '#34495e',       # 深灰色
        'text': '#2c3e50'         # 深蓝灰色
    }
    
    # 绘制SSM核心结构
    # 状态更新方程: h_t = A h_{t-1} + B x_t
    # 输出方程: y_t = C h_t
    
    # 定义组件位置和大小
    state_box = {'pos': (4, 4), 'width': 2, 'height': 2}
    input_box = {'pos': (1, 4), 'width': 1.5, 'height': 1}
    output_box = {'pos': (8, 4.5), 'width': 1.5, 'height': 1}
    
    # 绘制状态框
    state_rect = patches.Rectangle(
        state_box['pos'], 
        state_box['width'], state_box['height'],
        linewidth=2,
        edgecolor=colors['state'],
        facecolor=colors['state'] + '40',
        alpha=0.9,
        zorder=2
    )
    ax.add_patch(state_rect)
    ax.text(state_box['pos'][0] + state_box['width']/2, state_box['pos'][1] + state_box['height']/2, 
           '状态 h_t', ha='center', va='center', fontsize=14, color=colors['text'], fontweight='bold')
    
    # 绘制输入框
    input_rect = patches.Rectangle(
        input_box['pos'], 
        input_box['width'], input_box['height'],
        linewidth=2,
        edgecolor=colors['input'],
        facecolor=colors['input'] + '40',
        alpha=0.9,
        zorder=2
    )
    ax.add_patch(input_rect)
    ax.text(input_box['pos'][0] + input_box['width']/2, input_box['pos'][1] + input_box['height']/2, 
           '输入 x_t', ha='center', va='center', fontsize=14, color=colors['text'], fontweight='bold')
    
    # 绘制输出框
    output_rect = patches.Rectangle(
        output_box['pos'], 
        output_box['width'], output_box['height'],
        linewidth=2,
        edgecolor=colors['output'],
        facecolor=colors['output'] + '40',
        alpha=0.9,
        zorder=2
    )
    ax.add_patch(output_rect)
    ax.text(output_box['pos'][0] + output_box['width']/2, output_box['pos'][1] + output_box['height']/2, 
           '输出 y_t', ha='center', va='center', fontsize=14, color=colors['text'], fontweight='bold')
    
    # 绘制参数A
    ax.text(4.5, 2.5, 'A', ha='center', va='center', fontsize=16, color=colors['param'], fontweight='bold')
    
    # 绘制参数B
    ax.text(3, 4.5, 'B', ha='center', va='center', fontsize=16, color=colors['param'], fontweight='bold')
    
    # 绘制参数C
    ax.text(6.5, 5, 'C', ha='center', va='center', fontsize=16, color=colors['param'], fontweight='bold')
    
    # 绘制循环连接
    # 状态自循环 (A)
    arc = patches.Arc((4, 3), 1, 1, theta1=180, theta2=360, linewidth=2, color=colors['arrow'])
    ax.add_patch(arc)
    ax.arrow(4.5, 2.6, 0, 0.2, head_width=0.1, head_length=0.1, fc=colors['arrow'], ec=colors['arrow'])
    
    # 输入到状态 (B)
    ax.arrow(input_box['pos'][0] + input_box['width'], input_box['pos'][1] + input_box['height']/2,
            state_box['pos'][0] - (input_box['pos'][0] + input_box['width']), 0,
            head_width=0.2, head_length=0.2, fc=colors['arrow'], ec=colors['arrow'],
            length_includes_head=True, linewidth=2, zorder=1)
    
    # 状态到输出 (C)
    ax.arrow(state_box['pos'][0] + state_box['width'], state_box['pos'][1] + state_box['height']/2,
            output_box['pos'][0] - (state_box['pos'][0] + state_box['width']), 0,
            head_width=0.2, head_length=0.2, fc=colors['arrow'], ec=colors['arrow'],
            length_includes_head=True, linewidth=2, zorder=1)
    
    # 添加方程
    ax.text(5, 7, r'状态更新: $h_t = Ah_{t-1} + Bx_t$', ha='center', va='center', 
           fontsize=16, color=colors['text'])
    ax.text(5, 6.5, r'输出计算: $y_t = Ch_t$', ha='center', va='center', 
           fontsize=16, color=colors['text'])
    
    # 添加Mamba特有的选择性扫描机制说明
    selection_text = (
        "Mamba的创新: 选择性扫描机制\n"
        "• 参数A和B是输入依赖的: A(x), B(x)\n"
        "• 通过可学习的选择性机制决定保留哪些信息\n"
        "• 实现了高效的并行计算和长距离依赖建模"
    )
    
    # 添加文本框
    props = dict(boxstyle='round', facecolor='#f39c12', alpha=0.2)
    ax.text(5, 1.5, selection_text, ha='center', va='center', fontsize=14,
           color=colors['text'], bbox=props)
    
    # 设置坐标轴
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 添加标题
    plt.suptitle('状态空间模型(SSM)核心机制', fontsize=20, y=0.98)
    
    # 保存图形
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close()
    
    logging.info(f"SSM详细结构示意图已保存到: {output_path}")
    
    return output_path


def main():
    """主函数"""
    output_dir = os.path.join(os.getcwd(), "mamba_visualizations")
    
    # 创建Mamba架构示意图
    mamba_diagram_path = create_mamba_architecture_diagram(output_dir)
    
    # 创建SSM详细结构示意图
    ssm_diagram_path = create_ssm_detail_diagram(output_dir)
    
    logging.info("所有可视化图表已生成完成！")
    logging.info(f"输出目录: {output_dir}")
    
    return mamba_diagram_path, ssm_diagram_path


if __name__ == "__main__":
    main()