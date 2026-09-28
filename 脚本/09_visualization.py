# ============================================================
# 09_visualization.py
# 功能：高级可视化 - t-SNE降维对比 + 中国地图聚类空间分布
# 输入：数据/features/standardized_data.xlsx
#       数据/features/cluster_euclidean.xlsx（伪标签）
# 输出：图表/fig09_tsne_comparison.png
#       图表/fig10_cluster_map.png
# 依赖：cartopy（地图功能）, metric-learn（LMNN空间t-SNE）
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from config import (
    STANDARDIZED_DATA_PATH,
    CLUSTER_EUCLIDEAN_PATH,
    K_RANGE,
    RANDOM_SEED,
    NCA_MAX_ITER,
    LMNN_K,
    LMNN_LEARN_RATE,
    LMNN_MAX_ITER,
    CITIES,
    FIGURE_FOLDER,
    MAP_DATA_FOLDER,
    ensure_dirs
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入LMNN
LMNN_AVAILABLE = False
try:
    from metric_learn import LMNN
    LMNN_AVAILABLE = True
except ImportError:
    pass

# 尝试导入cartopy
CARTOPY_AVAILABLE = False
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    pass

# 城市坐标（经纬度）
CITY_COORDS = {
    '北京': (116.4, 39.9), '天津': (117.2, 39.13), '上海': (121.48, 31.22), '重庆': (106.54, 29.59),
    '石家庄': (114.48, 38.03), '太原': (112.56, 37.87), '呼和浩特': (111.65, 40.82),
    '沈阳': (123.43, 41.8), '长春': (125.35, 43.88), '哈尔滨': (126.63, 45.75),
    '南京': (118.78, 32.04), '杭州': (120.15, 30.28), '合肥': (117.27, 31.86),
    '福州': (119.3, 26.08), '南昌': (115.89, 28.68), '济南': (117.0, 36.65),
    '郑州': (113.65, 34.76), '武汉': (114.31, 30.52), '长沙': (112.98, 28.21),
    '广州': (113.23, 23.16), '南宁': (108.32, 22.84), '海口': (110.35, 20.02),
    '成都': (104.06, 30.67), '贵阳': (106.71, 26.57), '昆明': (102.73, 25.04),
    '拉萨': (91.14, 29.66), '西安': (108.95, 34.27), '兰州': (103.73, 36.03),
    '西宁': (101.74, 36.56), '银川': (106.27, 38.47), '乌鲁木齐': (87.68, 43.77),
}


def load_data():
    """加载标准化数据和伪标签"""
    if not os.path.exists(STANDARDIZED_DATA_PATH):
        print(f'[错误] 文件不存在：{STANDARDIZED_DATA_PATH}')
        print('  请先运行 03_feature_engineering.py')
        exit(1)

    df = pd.read_excel(STANDARDIZED_DATA_PATH)
    feature_cols = [col for col in df.columns if col not in ['城市', '年份']]
    X = df[feature_cols].values
    cities = df['城市'].values
    years = df['年份'].values

    # 获取伪标签
    if os.path.exists(CLUSTER_EUCLIDEAN_PATH):
        df_euc = pd.read_excel(CLUSTER_EUCLIDEAN_PATH)
        y_pseudo = df_euc['聚类标签'].values
    else:
        kmeans_temp = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
        y_pseudo = kmeans_temp.fit_predict(X)

    return df, X, feature_cols, cities, years, y_pseudo


def plot_tsne(df, X, feature_cols, years, y_pseudo):
    """t-SNE降维可视化三种空间"""
    print('\n--- t-SNE 降维可视化 ---')

    # 年份对应的标记和颜色
    year_markers = {2016: 'o', 2017: 's', 2018: '^', 2019: 'D'}
    cluster_colors = {0: '#1f77b4', 1: '#d62728'}

    # 1. 原始欧氏距离空间
    print('  计算 t-SNE (欧氏距离空间)...')
    tsne_euclidean = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)
    X_tsne_euclidean = tsne_euclidean.fit_transform(X)

    # 2. NCA变换空间
    print('  训练NCA并计算 t-SNE...')
    nca = NeighborhoodComponentsAnalysis(
        n_components=len(feature_cols),
        random_state=RANDOM_SEED,
        max_iter=NCA_MAX_ITER
    )
    X_nca = nca.fit_transform(X, y_pseudo)
    tsne_nca = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)
    X_tsne_nca = tsne_nca.fit_transform(X_nca)

    # 3. LMNN变换空间
    X_tsne_lmnn = None
    if LMNN_AVAILABLE:
        print('  训练LMNN并计算 t-SNE...')
        try:
            lmnn = LMNN(k=LMNN_K, learn_rate=LMNN_LEARN_RATE, max_iter=LMNN_MAX_ITER, verbose=False)
            X_lmnn = lmnn.fit_transform(X, y_pseudo)
            tsne_lmnn = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)
            X_tsne_lmnn = tsne_lmnn.fit_transform(X_lmnn)
        except Exception as e:
            print(f'  LMNN训练失败: {e}')

    # 绘制对比图
    n_plots = 3 if X_tsne_lmnn is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    def scatter_space(ax, X_tsne, title):
        for year in [2016, 2017, 2018, 2019]:
            year_mask = years == year
            for label in np.unique(y_pseudo):
                mask = (y_pseudo == label) & year_mask
                if mask.sum() > 0:
                    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                               c=[cluster_colors.get(label, 'gray')],
                               marker=year_markers[year],
                               s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('t-SNE 维度1')
        ax.set_ylabel('t-SNE 维度2')
        ax.grid(True, alpha=0.3)

    scatter_space(axes[0], X_tsne_euclidean, '(a) 欧氏距离空间')
    scatter_space(axes[1], X_tsne_nca, '(b) NCA变换空间')
    if X_tsne_lmnn is not None:
        scatter_space(axes[2], X_tsne_lmnn, '(c) LMNN变换空间')

    # 图例
    unique_labels = np.unique(y_pseudo)
    legend_elements = []
    for label in unique_labels:
        color = cluster_colors.get(label, 'gray')
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                   markersize=8, label=f'聚类{label}')
        )
    for year, marker in year_markers.items():
        legend_elements.append(
            Line2D([0], [0], marker=marker, color='k', markersize=8,
                   label=f'{year}年', markerfacecolor='w')
        )
    axes[0].legend(handles=legend_elements, loc='best', fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_FOLDER, 'fig09_tsne_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  t-SNE对比图已保存：{fig_path}')


def read_gmt_file(filename):
    """读取GMT格式的边界数据文件"""
    filepath = os.path.join(MAP_DATA_FOLDER, filename)
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = [block.strip() for block in content.split('>') if block.strip()]
    lines = []
    for block in blocks:
        points = []
        for row in block.strip().split('\n'):
            if row.strip() and not row.startswith('#'):
                parts = row.strip().split()
                if len(parts) >= 2:
                    try:
                        points.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
        if points:
            lines.append(points)
    return lines


def plot_cluster_map():
    """绘制中国城市聚类地理分布图"""
    print('\n--- 城市聚类地图 ---')

    if not CARTOPY_AVAILABLE:
        print('  [跳过] cartopy未安装，无法绘制地图')
        print('  安装命令：pip install cartopy')
        return

    # 检查边界文件
    border_lines = read_gmt_file('CN-border-La.gmt')
    if border_lines is None:
        print(f'  [跳过] 未找到边界文件：{MAP_DATA_FOLDER}/CN-border-La.gmt')
        print('  请下载中国地理边界数据放入地图数据/目录')
        return

    nine_lines = read_gmt_file('ten-dash-line.gmt')

    # 读取聚类结果
    if not os.path.exists(CLUSTER_EUCLIDEAN_PATH):
        print('  [跳过] 欧氏距离聚类结果不存在')
        return

    df_euclidean = pd.read_excel(CLUSTER_EUCLIDEAN_PATH)

    # 获取每个城市的主要聚类
    city_main_cluster = {}
    for city in CITIES:
        city_data = df_euclidean[df_euclidean['城市'] == city]
        if len(city_data) > 0:
            mean_label = city_data['聚类标签'].mean()
            main_cluster = 1 if mean_label > 0.5 else 0
            city_main_cluster[city] = main_cluster
        else:
            city_main_cluster[city] = -1

    # 绘图
    fig = plt.figure(figsize=(16, 12))

    proj_main = ccrs.LambertConformal(central_longitude=105, central_latitude=35)
    ax = fig.add_axes([0.08, 0.08, 0.65, 0.85], projection=proj_main)

    # 地图要素
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), alpha=0.1, color='lightblue')
    ax.add_feature(cfeature.LAND.with_scale('50m'), alpha=0.05, color='lightgray')
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=0.5, alpha=0.3)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), alpha=0.3, edgecolor='blue', facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)

    # 绘制国界线
    for bline in border_lines:
        xs = [p[0] for p in bline]
        ys = [p[1] for p in bline]
        ax.plot(xs, ys, '-', color='black', linewidth=0.8, transform=ccrs.Geodetic())

    # 绘制九段线
    if nine_lines:
        for nline in nine_lines:
            xs = [p[0] for p in nline]
            ys = [p[1] for p in nline]
            ax.plot(xs, ys, '--', color='black', linewidth=0.8, transform=ccrs.Geodetic())
        ax.text(116, 18, '南海诸岛', fontsize=10, ha='center', va='center',
                transform=ccrs.PlateCarree(), fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_extent([70, 135, 15, 55])

    # 网格线
    gl = ax.gridlines(linestyle='--', alpha=0.3, draw_labels=True, linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}

    # 颜色映射
    colors = {0: '#2ca02c', 1: '#d62728'}

    # 绘制城市散点
    for cname, clabel in city_main_cluster.items():
        if cname in CITY_COORDS and clabel != -1:
            cx, cy = CITY_COORDS[cname]
            ax.scatter(cx, cy, c=colors[clabel], s=150,
                       edgecolors='black', linewidth=1.2, alpha=0.9,
                       transform=ccrs.PlateCarree(), zorder=10)
            ax.text(cx, cy + 0.3, cname, transform=ccrs.PlateCarree(),
                    fontsize=7, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7))

    # 图例
    ax.legend(handles=[Patch(facecolor=colors[0], label='清洁城市'),
                       Patch(facecolor=colors[1], label='重污染城市')],
              loc='lower left', fontsize=11)
    ax.set_title('中国城市空气质量聚类空间分布（2016-2019）', fontsize=16, pad=15)

    # 南海小图
    proj_south = ccrs.LambertConformal(central_longitude=115, central_latitude=15)
    sub_ax = fig.add_axes([0.75, 0.12, 0.2, 0.25], projection=proj_south)
    sub_ax.add_feature(cfeature.OCEAN.with_scale('50m'), alpha=0.1, color='lightblue')
    sub_ax.add_feature(cfeature.LAND.with_scale('50m'), alpha=0.05, color='lightgray')
    sub_ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)

    for bline in border_lines:
        xs = [p[0] for p in bline]
        ys = [p[1] for p in bline]
        sub_ax.plot(xs, ys, '-', color='black', linewidth=0.5, transform=ccrs.Geodetic())

    if nine_lines:
        for nline in nine_lines:
            xs = [p[0] for p in nline]
            ys = [p[1] for p in nline]
            sub_ax.plot(xs, ys, '--', color='black', linewidth=0.5, transform=ccrs.Geodetic())

    sub_ax.set_extent([105, 125, 0, 25])
    sub_ax.scatter(110.35, 20.02, c='black', s=30, transform=ccrs.PlateCarree())
    sub_ax.text(116, 16, '南海诸岛', fontsize=6, ha='center', va='center',
                transform=ccrs.PlateCarree(),
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
    sub_ax.set_title('南海诸岛', fontsize=8)

    # 保存
    fig_path = os.path.join(FIGURE_FOLDER, 'fig10_cluster_map.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  聚类地图已保存：{fig_path}')


def main():
    ensure_dirs()

    print('=' * 50)
    print('09 高级可视化')
    print('=' * 50)

    # 加载数据
    df, X, feature_cols, cities, years, y_pseudo = load_data()
    print(f'数据形状：{X.shape}')

    # 1. t-SNE可视化
    plot_tsne(df, X, feature_cols, years, y_pseudo)

    # 2. 聚类地图
    plot_cluster_map()

    print('\n' + '=' * 50)
    print('可视化完成！')


if __name__ == '__main__':
    main()
