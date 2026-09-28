# ============================================================
# 06_nca_experiment.py
# 功能：NCA度量学习 + K-means 聚类实验
# 输入：数据/features/standardized_data.xlsx
#       数据/features/cluster_euclidean.xlsx（用作伪标签来源）
# 输出：results/nca_scores.csv
#       图表/fig05_nca_comparison.png
#       数据/features/cluster_nca.xlsx
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from config import (
    STANDARDIZED_DATA_PATH,
    CLUSTER_EUCLIDEAN_PATH,
    CLUSTER_NCA_PATH,
    K_RANGE,
    RANDOM_SEED,
    NCA_MAX_ITER,
    YEARS,
    FIGURE_FOLDER,
    RESULTS_FOLDER,
    ensure_dirs
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    ensure_dirs()

    print('=' * 50)
    print('06 NCA度量学习 + K-means 聚类')
    print('=' * 50)

    # 读取标准化数据
    if not os.path.exists(STANDARDIZED_DATA_PATH):
        print(f'[错误] 文件不存在：{STANDARDIZED_DATA_PATH}')
        print('  请先运行 03_feature_engineering.py')
        exit(1)

    df = pd.read_excel(STANDARDIZED_DATA_PATH)

    # 特征列（排除城市和年份）
    feature_cols = [col for col in df.columns if col not in ['城市', '年份']]
    X = df[feature_cols].values

    print(f'数据形状：{X.shape}')
    print(f'特征数量：{len(feature_cols)}')

    # ----------------------------------------------------------
    # 获取伪标签（来自欧氏距离聚类结果）
    # ----------------------------------------------------------
    if not os.path.exists(CLUSTER_EUCLIDEAN_PATH):
        print(f'[警告] 欧氏距离聚类结果不存在，使用临时K-means生成伪标签...')
        kmeans_temp = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
        y_pseudo = kmeans_temp.fit_predict(X)
    else:
        df_euclidean = pd.read_excel(CLUSTER_EUCLIDEAN_PATH)
        y_pseudo = df_euclidean['聚类标签'].values

    print(f'伪标签类别数：{len(np.unique(y_pseudo))}')

    # ----------------------------------------------------------
    # NCA训练（保持原始维度，不降维）
    # ----------------------------------------------------------
    print('\n--- 训练NCA ---')
    try:
        nca = NeighborhoodComponentsAnalysis(
            n_components=len(feature_cols),  # 保持23维，不降维
            random_state=RANDOM_SEED,
            max_iter=NCA_MAX_ITER,
            verbose=0
        )
        X_nca = nca.fit_transform(X, y_pseudo)
        print(f'NCA训练完成！新空间维度：{X_nca.shape[1]}')
    except Exception as e:
        print(f'[错误] NCA训练失败：{e}')
        exit(1)

    # ----------------------------------------------------------
    # 在NCA空间中进行K-means聚类
    # ----------------------------------------------------------
    print('\n--- NCA空间K-means聚类 ---')
    k_range = list(K_RANGE)
    silhouette_scores_nca = []
    db_scores_nca = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(X_nca)

        sil = silhouette_score(X_nca, labels)
        db = davies_bouldin_score(X_nca, labels)
        silhouette_scores_nca.append(sil)
        db_scores_nca.append(db)

        print(f'  k={k}: 轮廓系数={sil:.4f}, DB指数={db:.4f}')

    # 保存评估结果CSV
    scores_df = pd.DataFrame({
        'k': k_range,
        'silhouette': silhouette_scores_nca,
        'davies_bouldin': db_scores_nca
    })
    csv_path = os.path.join(RESULTS_FOLDER, 'nca_scores.csv')
    scores_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\n  评估结果已保存：{csv_path}')

    # ----------------------------------------------------------
    # 选择最佳k，执行最终聚类
    # ----------------------------------------------------------
    best_k_nca = k_range[np.argmax(silhouette_scores_nca)]
    best_sil_nca = max(silhouette_scores_nca)
    print(f'\n  最佳聚类数: k={best_k_nca}, 轮廓系数={best_sil_nca:.4f}')

    kmeans_nca = KMeans(n_clusters=best_k_nca, random_state=RANDOM_SEED, n_init=10)
    labels_nca = kmeans_nca.fit_predict(X_nca)

    # 保存NCA聚类结果
    df_nca_result = df.copy()
    df_nca_result['聚类标签_NCA'] = labels_nca
    df_nca_result.to_excel(CLUSTER_NCA_PATH, index=False)
    print(f'  聚类结果已保存：{CLUSTER_NCA_PATH}')

    # 显示各聚类城市分布
    print('\nNCA各聚类城市分布：')
    for cluster_id in range(best_k_nca):
        cluster_cities = df_nca_result[df_nca_result['聚类标签_NCA'] == cluster_id]
        print(f'\n  聚类 {cluster_id} (共{len(cluster_cities)}个样本):')
        for year in YEARS:
            year_cities = cluster_cities[cluster_cities['年份'] == year]['城市'].tolist()
            if year_cities:
                print(f'    {year}年: {", ".join(year_cities)}')

    # ----------------------------------------------------------
    # 对比图：NCA vs 欧氏距离
    # ----------------------------------------------------------
    # 计算欧氏距离基准（原始空间）
    labels_euclidean_base = KMeans(n_clusters=best_k_nca, random_state=RANDOM_SEED, n_init=10).fit_predict(X)
    sil_euclidean = silhouette_score(X, labels_euclidean_base)
    db_euclidean = davies_bouldin_score(X, labels_euclidean_base)

    print(f'\n--- 对比 ---')
    print(f'  欧氏距离 (k={best_k_nca}): 轮廓系数={sil_euclidean:.4f}, DB指数={db_euclidean:.4f}')
    print(f'  NCA      (k={best_k_nca}): 轮廓系数={best_sil_nca:.4f}, DB指数={db_scores_nca[best_k_nca - 2]:.4f}')

    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(k_range, silhouette_scores_nca, 'gs-', linewidth=2, markersize=8, label='NCA')
    axes[0].axhline(y=sil_euclidean, color='b', linestyle='--', linewidth=2, label=f'欧氏距离 (k={best_k_nca})')
    axes[0].set_xlabel('聚类数 k', fontsize=12)
    axes[0].set_ylabel('轮廓系数', fontsize=12)
    axes[0].set_title('NCA vs 欧氏距离 - 轮廓系数', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, db_scores_nca, 'gs-', linewidth=2, markersize=8, label='NCA')
    axes[1].axhline(y=db_euclidean, color='b', linestyle='--', linewidth=2, label=f'欧氏距离 (k={best_k_nca})')
    axes[1].set_xlabel('聚类数 k', fontsize=12)
    axes[1].set_ylabel('DB指数', fontsize=12)
    axes[1].set_title('NCA vs 欧氏距离 - DB指数', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_FOLDER, 'fig05_nca_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  对比图已保存：{fig_path}')

    print('\n' + '=' * 50)
    print('NCA实验完成！')


if __name__ == '__main__':
    main()
