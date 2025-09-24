#!/usr/bin/env python3
"""
基于all_results.csv数据绘制增强版图表
包含误差条和加权线性回归的置信带
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import seaborn as sns

def load_and_process_data(csv_file):
    """加载并处理数据"""
    df = pd.read_csv(csv_file)
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print(f"前几行数据:\n{df.head()}")
    
    # 移除空行和无效数据
    df = df.dropna(subset=['depth', 'best_lr'])
    df = df[df['best_lr'] > 0]  # 确保学习率为正数
    
    print(f"清理后数据形状: {df.shape}")
    
    # 计算log10(η*)
    df['log10_lr'] = np.log10(df['best_lr'])
    df['log10_depth'] = np.log10(df['depth'])
    
    return df

def calculate_error_bars(df):
    """计算每个深度的误差条（均值±95% CI）"""
    # 按深度分组计算统计量
    grouped = df.groupby('depth')['log10_lr'].agg(['mean', 'std', 'count']).reset_index()
    
    # 计算95%置信区间
    # 使用t分布，因为样本量可能较小
    confidence_level = 0.95
    grouped['ci_95'] = stats.t.ppf(1 - (1 - confidence_level) / 2, grouped['count'] - 1) * grouped['std'] / np.sqrt(grouped['count'])
    
    # 处理标准差为0的情况（只有一个数据点）
    grouped['ci_95'] = grouped['ci_95'].fillna(0)
    
    return grouped

def weighted_linear_regression(x, y, weights=None):
    """执行加权线性回归"""
    if weights is None:
        weights = np.ones(len(x))
    
    # 使用sklearn的LinearRegression，通过样本权重实现加权回归
    reg = LinearRegression()
    reg.fit(x.reshape(-1, 1), y, sample_weight=weights)
    
    # 计算预测值和残差
    y_pred = reg.predict(x.reshape(-1, 1))
    residuals = y - y_pred
    
    # 计算加权残差平方和
    weighted_ss_res = np.sum(weights * residuals**2)
    weighted_ss_tot = np.sum(weights * (y - np.average(y, weights=weights))**2)
    
    # 计算R²
    r_squared = 1 - (weighted_ss_res / weighted_ss_tot)
    
    return reg, y_pred, r_squared

def calculate_confidence_band(x, y, y_pred, confidence_level=0.95):
    """计算回归线的置信带"""
    n = len(x)
    residuals = y - y_pred
    mse = np.mean(residuals**2)
    
    # 计算标准误差
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean)**2)
    
    # 预测点的标准误差
    se_pred = np.sqrt(mse * (1/n + (x - x_mean)**2 / sxx))
    
    # t值
    t_val = stats.t.ppf(1 - (1 - confidence_level) / 2, n - 2)
    
    # 置信带
    margin = t_val * se_pred
    upper_bound = y_pred + margin
    lower_bound = y_pred - margin
    
    return upper_bound, lower_bound

def plot_enhanced_results(csv_file, save_path=None):
    """绘制增强版结果图表"""
    # 加载数据
    df = load_and_process_data(csv_file)
    print(f"加载了 {len(df)} 个数据点")
    
    # 计算误差条
    error_data = calculate_error_bars(df)
    print(f"计算了 {len(error_data)} 个深度的误差条")
    
    # 准备回归数据
    x = df['log10_depth'].values
    y = df['log10_lr'].values
    
    # 使用损失作为权重（损失越小，权重越大）
    weights = 1 / (df['best_loss'].values + 1e-8)  # 避免除零
    weights = weights / np.sum(weights) * len(weights)  # 归一化
    
    # 执行加权线性回归
    reg, y_pred, r_squared = weighted_linear_regression(x, y, weights)
    
    # 计算置信带
    upper_bound, lower_bound = calculate_confidence_band(x, y, y_pred)
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 绘制原始数据点
    ax.scatter(df['depth'], df['best_lr'], alpha=0.6, s=50, color='blue', label='Grid Search')
    
    # 绘制误差条
    for _, row in error_data.iterrows():
        depth = row['depth']
        mean_lr = 10**row['mean']  # 转换回原始尺度
        ci_95 = row['ci_95']
        
        # 计算误差条的上下限
        lower_err = mean_lr - 10**(row['mean'] - ci_95)
        upper_err = 10**(row['mean'] + ci_95) - mean_lr
        
        ax.errorbar(depth, mean_lr, yerr=[[lower_err], [upper_err]], 
                   fmt='o', color='red', capsize=5, capthick=2, 
                   markersize=8, alpha=0.8, label='Mean ± 95% CI' if depth == error_data['depth'].iloc[0] else "")
    
    # 绘制加权回归线
    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = reg.predict(x_range.reshape(-1, 1))
    
    # 转换回原始尺度
    x_range_orig = 10**x_range
    y_range_orig = 10**y_range
    
    # 创建详细的图例标签
    fit_label = f'Weighted Fit: η ∝ L^({reg.coef_[0]:.3f})'
    stats_label = f'Slope: {reg.coef_[0]:.4f}, Intercept: {reg.intercept_:.4f}, R²: {r_squared:.4f}'
    
    ax.plot(x_range_orig, y_range_orig, 'r--', linewidth=2, label=fit_label)
    
    # 绘制置信带
    upper_bound_orig = 10**upper_bound
    lower_bound_orig = 10**lower_bound
    x_orig = 10**x
    
    # 按x排序以正确绘制填充区域
    sort_idx = np.argsort(x_orig)
    x_sorted = x_orig[sort_idx]
    upper_sorted = upper_bound_orig[sort_idx]
    lower_sorted = lower_bound_orig[sort_idx]
    
    ax.fill_between(x_sorted, lower_sorted, upper_sorted, 
                   alpha=0.2, color='red', label='95% Confidence Band')
    
    # 绘制理论线
    k_theory = df['best_lr'].iloc[0] * (df['depth'].iloc[0] ** 1.5)
    theory_line = k_theory * (x_range_orig ** (-1.5))
    theory_label = f'Theory: η ∝ L^(-1.5)'
    ax.plot(x_range_orig, theory_line, 'g-.', linewidth=2, label=theory_label)
    
    # 添加统计信息到图例中
    diff_from_theory = abs(reg.coef_[0] - (-1.5))
    stats_legend_label = f'Stats: Slope={reg.coef_[0]:.4f}, Intercept={reg.intercept_:.4f}, R²={r_squared:.4f}, Diff={diff_from_theory:.4f}'
    
    # 创建一个不可见的线条来添加统计信息到图例
    ax.plot([], [], ' ', label=stats_legend_label)
    
    # 设置坐标轴
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Depth L')
    ax.set_ylabel('Optimal Learning Rate η*')
    ax.set_title('Global Power Law Fit\nwith Error Bars and Weighted Regression')
    
    # 添加网格和图例
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()
    
    # 打印详细统计信息
    print(f"\n{'='*60}")
    print("详细统计信息:")
    print(f"{'='*60}")
    print(f"加权回归斜率: {reg.coef_[0]:.4f}")
    print(f"理论斜率: -1.5000")
    print(f"斜率差异: {abs(reg.coef_[0] - (-1.5)):.4f}")
    print(f"R²: {r_squared:.4f}")
    print(f"数据点数量: {len(df)}")
    print(f"深度范围: {df['depth'].min()} - {df['depth'].max()}")
    print(f"学习率范围: {df['best_lr'].min():.6f} - {df['best_lr'].max():.6f}")
    
    return fig, reg, r_squared

if __name__ == "__main__":
    fig, reg, r_squared = plot_enhanced_results(
        'all_results.csv', 
        save_path='enhanced_power_law_fit.png'
    )
