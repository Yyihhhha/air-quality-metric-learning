# ============================================================
# 04_eda_analysis.py
# 功能：污染物关联分析 - 相关性热力图 + 季节差异分析
# 输入：数据/processed/{城市}_{年份}_daily.xlsx
# 输出：图表/fig01_correlation_heatmap.png
#       图表/fig02_seasonal_comparison.png
#       图表/fig03_pm25_boxplot.png
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    PROCESSED_DATA_FOLDER,
    FIGURE_FOLDER,
    CITIES,
    YEARS,
    ensure_dirs
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义污染物列表
POLLUTANTS = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'AQI']
POLLUTANTS_NO_AQI = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']


def get_season(date):
    """根据月份返回季节"""
    month = date.month
    if month in [3, 4, 5]:
        return '春季'
    elif month in [6, 7, 8]:
        return '夏季'
    elif month in [9, 10, 11]:
        return '秋季'
    else:
        return '冬季'


def load_daily_data():
    """读取所有城市所有年份的逐日数据并合并"""
    daily_data_list = []

    for city in CITIES:
        for year in YEARS:
            filename = f"{city}_{year}_daily.xlsx"
            filepath = os.path.join(PROCESSED_DATA_FOLDER, filename)

            if not os.path.exists(filepath):
                continue

            try:
                df = pd.read_excel(filepath)
                df['date'] = pd.to_datetime(df['date'])
                df['城市'] = city
                df['年份'] = year
                df['季节'] = df['date'].apply(get_season)
                daily_data_list.append(df)
            except Exception as e:
                print(f"  [警告] 读取失败：{filename} - {e}")

    if not daily_data_list:
        print("  [错误] 未找到任何逐日数据文件！")
        print("  请先运行 01_data_preprocessing.py")
        exit(1)

    df_all = pd.concat(daily_data_list, ignore_index=True)
    return df_all


def plot_correlation_heatmap(corr_matrix):
    """绘制相关系数热力图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5,
                vmin=-0.5, vmax=1, ax=ax)
    ax.set_title('六项污染物相关系数热力图（2016-2019）', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(FIGURE_FOLDER, 'fig01_correlation_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  热力图已保存：{output_path}")


def plot_seasonal_comparison(corr_winter, corr_summer):
    """绘制冬夏相关系数对比柱状图"""
    pairs = [('PM2.5', 'CO'), ('PM2.5', 'O3'), ('SO2', 'PM2.5'), ('PM2.5', 'NO2')]

    seasonal_comparison = []
    for p1, p2 in pairs:
        r_winter = corr_winter.loc[p1, p2]
        r_summer = corr_summer.loc[p1, p2]
        seasonal_comparison.append({
            '污染物对': f"{p1}-{p2}",
            '冬季相关系数': round(r_winter, 4),
            '夏季相关系数': round(r_summer, 4),
        })

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(seasonal_comparison))
    width = 0.35

    winter_vals = [item['冬季相关系数'] for item in seasonal_comparison]
    summer_vals = [item['夏季相关系数'] for item in seasonal_comparison]
    pair_labels = [item['污染物对'] for item in seasonal_comparison]

    bars1 = ax.bar(x - width / 2, winter_vals, width, label='冬季', color='#1f77b4')
    bars2 = ax.bar(x + width / 2, summer_vals, width, label='夏季', color='#d62728')

    ax.set_ylabel('相关系数', fontsize=12)
    ax.set_xlabel('污染物对', fontsize=12)
    ax.set_title('冬夏两季污染物相关系数对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(FIGURE_FOLDER, 'fig02_seasonal_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  冬夏对比图已保存：{output_path}")

    return seasonal_comparison


def plot_pm25_boxplot(df_all):
    """绘制PM2.5季节箱线图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    season_order = ['冬季', '春季', '夏季', '秋季']
    season_data = []
    for season in season_order:
        values = df_all[df_all['季节'] == season]['PM2.5'].dropna()
        # 去除极端异常值（>99分位数）
        q99 = values.quantile(0.99)
        values = values[values <= q99]
        season_data.append(values)

    bp = ax.boxplot(season_data, labels=season_order, patch_artist=True,
                    boxprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='red'),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('PM2.5浓度 (μg/m³)', fontsize=12)
    ax.set_xlabel('季节', fontsize=12)
    ax.set_title('PM2.5浓度季节分布', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(FIGURE_FOLDER, 'fig03_pm25_boxplot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  季节箱线图已保存：{output_path}")


def main():
    ensure_dirs()

    print('=' * 50)
    print('04 探索性数据分析（污染物关联）')
    print('=' * 50)

    # 加载数据
    print("\n读取逐日数据...")
    df_all = load_daily_data()
    print(f"  共 {len(df_all)} 条逐日记录")
    print(f"  涉及城市：{df_all['城市'].nunique()} 个")

    # 1. 全时段相关系数矩阵
    print("\n--- 计算相关系数矩阵 ---")
    df_pollutants = df_all[POLLUTANTS_NO_AQI].dropna()
    corr_matrix = df_pollutants.corr(method='pearson')

    print("关键相关系数：")
    print(f"  PM2.5-PM10: {corr_matrix.loc['PM2.5', 'PM10']:.4f}")
    print(f"  PM2.5-CO:   {corr_matrix.loc['PM2.5', 'CO']:.4f}")
    print(f"  PM2.5-O3:   {corr_matrix.loc['PM2.5', 'O3']:.4f}")

    # 2. 季节对比
    print("\n--- 冬夏季节相关系数对比 ---")
    df_winter = df_all[df_all['季节'] == '冬季']
    df_summer = df_all[df_all['季节'] == '夏季']
    corr_winter = df_winter[POLLUTANTS_NO_AQI].corr(method='pearson')
    corr_summer = df_summer[POLLUTANTS_NO_AQI].corr(method='pearson')

    # 3. PM2.5季节统计
    print("\n--- PM2.5浓度季节特征 ---")
    pm25_overall_mean = df_all['PM2.5'].mean()
    pm25_overall_std = df_all['PM2.5'].std()
    print(f"  全时段PM2.5均值：{pm25_overall_mean:.1f} +/- {pm25_overall_std:.1f} μg/m³")

    seasonal_means = df_all.groupby('季节')['PM2.5'].agg(['mean', 'std']).round(1)
    for season in ['冬季', '春季', '夏季', '秋季']:
        if season in seasonal_means.index:
            print(f"  {season}: {seasonal_means.loc[season, 'mean']:.1f} μg/m³")

    # 4. 生成图表
    print("\n--- 生成可视化图表 ---")
    plot_correlation_heatmap(corr_matrix)
    plot_seasonal_comparison(corr_winter, corr_summer)
    plot_pm25_boxplot(df_all)

    print("\n" + '=' * 50)
    print('探索性数据分析完成！')


if __name__ == '__main__':
    main()
