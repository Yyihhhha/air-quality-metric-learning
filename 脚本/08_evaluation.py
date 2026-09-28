# ============================================================
# 08_evaluation.py
# 功能：综合评估 - 内部指标对比 + 外部指标(ARI/AMI) + 训练时间
# 输入：results/baseline_scores.csv, results/nca_scores.csv, results/lmnn_scores.csv
#       数据/features/cluster_euclidean.xlsx, cluster_nca.xlsx, cluster_lmnn.xlsx
#       数据/features/standardized_data.xlsx
# 输出：图表/fig07_three_methods_compare.png
#       图表/fig08_training_time.png
#       results/external_evaluation.csv
#       results/training_time.csv
# ============================================================
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    adjusted_rand_score, adjusted_mutual_info_score
)
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from config import (
    STANDARDIZED_DATA_PATH,
    CLUSTER_EUCLIDEAN_PATH,
    CLUSTER_NCA_PATH,
    CLUSTER_LMNN_PATH,
    K_RANGE,
    RANDOM_SEED,
    NCA_MAX_ITER,
    LMNN_K,
    LMNN_LEARN_RATE,
    LMNN_MAX_ITER,
    CITY_TO_REGION,
    REGION_NAMES,
    FIGURE_FOLDER,
    RESULTS_FOLDER,
    ensure_dirs
)

warnings.filterwarnings('ignore')

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


def plot_three_methods_comparison():
    """读取三个CSV，绘制三种方法对比图"""
    print('\n--- 内部指标对比 ---')

    baseline_path = os.path.join(RESULTS_FOLDER, 'baseline_scores.csv')
    nca_path = os.path.join(RESULTS_FOLDER, 'nca_scores.csv')
    lmnn_path = os.path.join(RESULTS_FOLDER, 'lmnn_scores.csv')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 读取并绘制轮廓系数
    if os.path.exists(baseline_path):
        df_base = pd.read_csv(baseline_path)
        axes[0].plot(df_base['k'], df_base['silhouette'], 'bo-', linewidth=2, markersize=8, label='欧氏距离')
        axes[1].plot(df_base['k'], df_base['davies_bouldin'], 'bo-', linewidth=2, markersize=8, label='欧氏距离')
        print(f'  欧氏距离: 最佳轮廓系数={df_base["silhouette"].max():.4f} (k={df_base.loc[df_base["silhouette"].idxmax(), "k"]:.0f})')

    if os.path.exists(nca_path):
        df_nca = pd.read_csv(nca_path)
        axes[0].plot(df_nca['k'], df_nca['silhouette'], 'gs-', linewidth=2, markersize=8, label='NCA')
        axes[1].plot(df_nca['k'], df_nca['davies_bouldin'], 'gs-', linewidth=2, markersize=8, label='NCA')
        print(f'  NCA:      最佳轮廓系数={df_nca["silhouette"].max():.4f} (k={df_nca.loc[df_nca["silhouette"].idxmax(), "k"]:.0f})')

    if os.path.exists(lmnn_path):
        df_lmnn = pd.read_csv(lmnn_path)
        axes[0].plot(df_lmnn['k'], df_lmnn['silhouette'], 'r^-', linewidth=2, markersize=8, label='LMNN')
        axes[1].plot(df_lmnn['k'], df_lmnn['davies_bouldin'], 'r^-', linewidth=2, markersize=8, label='LMNN')
        print(f'  LMNN:     最佳轮廓系数={df_lmnn["silhouette"].max():.4f} (k={df_lmnn.loc[df_lmnn["silhouette"].idxmax(), "k"]:.0f})')

    axes[0].set_xlabel('聚类数 k', fontsize=12)
    axes[0].set_ylabel('轮廓系数', fontsize=12)
    axes[0].set_title('三种方法轮廓系数对比', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('聚类数 k', fontsize=12)
    axes[1].set_ylabel('DB指数', fontsize=12)
    axes[1].set_title('三种方法DB指数对比', fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_FOLDER, 'fig07_three_methods_compare.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  对比图已保存：{fig_path}')


def compute_external_evaluation():
    """计算外部评估指标 ARI/AMI（以地理分区为真实标签）"""
    print('\n--- 外部评估指标 (ARI/AMI) ---')

    if not os.path.exists(STANDARDIZED_DATA_PATH):
        print('  [跳过] 标准化数据不存在')
        return

    df_standard = pd.read_excel(STANDARDIZED_DATA_PATH)
    df_standard['key'] = df_standard['城市'] + '_' + df_standard['年份'].astype(str)
    df_standard_sorted = df_standard.sort_values('key').reset_index(drop=True)

    # 生成地理标签
    true_labels = np.array([CITY_TO_REGION.get(city, -1) for city in df_standard_sorted['城市'].values])

    if -1 in true_labels:
        unmapped = df_standard_sorted['城市'][true_labels == -1].unique().tolist()
        print(f'  [警告] 未映射的城市：{unmapped}')

    print(f'  样本总数：{len(true_labels)}')
    print(f'  真实类别数：{len(np.unique(true_labels))}')

    results = []

    # 辅助函数：对齐数据
    def align_labels(df_cluster):
        df_cluster['key'] = df_cluster['城市'] + '_' + df_cluster['年份'].astype(str)
        return df_cluster.sort_values('key').reset_index(drop=True)

    # 方法1：欧氏距离
    if os.path.exists(CLUSTER_EUCLIDEAN_PATH):
        df_euc = pd.read_excel(CLUSTER_EUCLIDEAN_PATH)
        df_euc_sorted = align_labels(df_euc)
        labels_euc = df_euc_sorted['聚类标签'].values
        ari = adjusted_rand_score(true_labels, labels_euc)
        ami = adjusted_mutual_info_score(true_labels, labels_euc)
        results.append({'方法': '欧氏距离 + K-means', 'ARI': round(ari, 4), 'AMI': round(ami, 4)})
        print(f'  欧氏距离: ARI={ari:.4f}, AMI={ami:.4f}')

    # 方法2：NCA
    if os.path.exists(CLUSTER_NCA_PATH):
        df_nca = pd.read_excel(CLUSTER_NCA_PATH)
        df_nca_sorted = align_labels(df_nca)
        labels_nca = df_nca_sorted['聚类标签_NCA'].values
        ari = adjusted_rand_score(true_labels, labels_nca)
        ami = adjusted_mutual_info_score(true_labels, labels_nca)
        results.append({'方法': 'NCA + K-means', 'ARI': round(ari, 4), 'AMI': round(ami, 4)})
        print(f'  NCA:      ARI={ari:.4f}, AMI={ami:.4f}')

    # 方法3：LMNN
    if os.path.exists(CLUSTER_LMNN_PATH):
        df_lmnn = pd.read_excel(CLUSTER_LMNN_PATH)
        df_lmnn_sorted = align_labels(df_lmnn)
        labels_lmnn = df_lmnn_sorted['聚类标签_LMNN'].values
        ari = adjusted_rand_score(true_labels, labels_lmnn)
        ami = adjusted_mutual_info_score(true_labels, labels_lmnn)
        results.append({'方法': 'LMNN + K-means', 'ARI': round(ari, 4), 'AMI': round(ami, 4)})
        print(f'  LMNN:     ARI={ari:.4f}, AMI={ami:.4f}')

    # 保存结果
    if results:
        eval_df = pd.DataFrame(results)
        eval_path = os.path.join(RESULTS_FOLDER, 'external_evaluation.csv')
        eval_df.to_csv(eval_path, index=False, encoding='utf-8-sig')
        print(f'\n  外部评估结果已保存：{eval_path}')


def measure_training_time():
    """测量三种方法训练时间（重复5次取平均）"""
    print('\n--- 训练时间测量 ---')

    if not os.path.exists(STANDARDIZED_DATA_PATH):
        print('  [跳过] 标准化数据不存在')
        return

    df = pd.read_excel(STANDARDIZED_DATA_PATH)
    feature_cols = [col for col in df.columns if col not in ['城市', '年份']]
    X = df[feature_cols].values

    # 获取伪标签
    if os.path.exists(CLUSTER_EUCLIDEAN_PATH):
        df_euc = pd.read_excel(CLUSTER_EUCLIDEAN_PATH)
        y_pseudo = df_euc['聚类标签'].values
    else:
        kmeans_temp = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
        y_pseudo = kmeans_temp.fit_predict(X)

    N_RUNS = 5
    print(f'  重复次数：{N_RUNS}')
    print(f'  数据形状：{X.shape}')

    # 1. 欧氏距离 + K-means
    times_euclidean = []
    for i in range(N_RUNS):
        start = time.time()
        kmeans = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
        kmeans.fit(X)
        times_euclidean.append(time.time() - start)

    # 2. NCA + K-means
    times_nca = []
    for i in range(N_RUNS):
        start = time.time()
        try:
            nca = NeighborhoodComponentsAnalysis(
                n_components=len(feature_cols),
                random_state=RANDOM_SEED,
                max_iter=NCA_MAX_ITER,
                verbose=0
            )
            X_nca = nca.fit_transform(X, y_pseudo)
            kmeans = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
            kmeans.fit(X_nca)
            times_nca.append(time.time() - start)
        except Exception:
            pass

    # 3. LMNN + K-means
    times_lmnn = []
    if LMNN_AVAILABLE:
        for i in range(N_RUNS):
            start = time.time()
            try:
                lmnn = LMNN(k=LMNN_K, learn_rate=LMNN_LEARN_RATE, max_iter=LMNN_MAX_ITER, verbose=False)
                X_lmnn = lmnn.fit_transform(X, y_pseudo)
                kmeans = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
                kmeans.fit(X_lmnn)
                times_lmnn.append(time.time() - start)
            except Exception:
                pass

    # 汇总
    time_results = []
    methods = []
    mean_times = []
    std_times = []

    if times_euclidean:
        t = np.mean(times_euclidean)
        s = np.std(times_euclidean)
        time_results.append({'方法': '欧氏距离+K-means', '平均时间(秒)': round(t, 6), '标准差': round(s, 6)})
        methods.append('欧氏距离')
        mean_times.append(t)
        std_times.append(s)
        print(f'  欧氏距离+K-means: {t:.6f} +/- {s:.6f} 秒')

    if times_nca:
        t = np.mean(times_nca)
        s = np.std(times_nca)
        time_results.append({'方法': 'NCA+K-means', '平均时间(秒)': round(t, 6), '标准差': round(s, 6)})
        methods.append('NCA')
        mean_times.append(t)
        std_times.append(s)
        print(f'  NCA+K-means:      {t:.6f} +/- {s:.6f} 秒')

    if times_lmnn:
        t = np.mean(times_lmnn)
        s = np.std(times_lmnn)
        time_results.append({'方法': 'LMNN+K-means', '平均时间(秒)': round(t, 6), '标准差': round(s, 6)})
        methods.append('LMNN')
        mean_times.append(t)
        std_times.append(s)
        print(f'  LMNN+K-means:     {t:.6f} +/- {s:.6f} 秒')

    # 保存CSV
    if time_results:
        time_df = pd.DataFrame(time_results)
        time_path = os.path.join(RESULTS_FOLDER, 'training_time.csv')
        time_df.to_csv(time_path, index=False, encoding='utf-8-sig')
        print(f'  训练时间结果已保存：{time_path}')

    # 绘制柱状图
    if methods:
        colors = ['#1f77b4', '#2ca02c', '#d62728'][:len(methods)]
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(methods, mean_times, yerr=std_times, capsize=5,
                      color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('方法', fontsize=12)
        ax.set_ylabel('训练时间（秒）', fontsize=12)
        ax.set_title('三种方法训练时间对比', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')

        for j, v in enumerate(mean_times):
            ax.text(j, v + 0.01 * max(mean_times), f'{v:.4f}',
                    ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        fig_path = os.path.join(FIGURE_FOLDER, 'fig08_training_time.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  训练时间图已保存：{fig_path}')


def main():
    ensure_dirs()

    print('=' * 50)
    print('08 综合评估')
    print('=' * 50)

    # 1. 内部指标对比（读取CSV）
    plot_three_methods_comparison()

    # 2. 外部评估指标
    compute_external_evaluation()

    # 3. 训练时间
    measure_training_time()

    print('\n' + '=' * 50)
    print('综合评估完成！')


if __name__ == '__main__':
    main()
