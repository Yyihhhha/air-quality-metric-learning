# ============================================================
# 05_baseline_kmeans.py
# 功能：欧氏距离 + K-means 基准聚类（含k值评估）
# 输入：数据/features/standardized_data.xlsx
# 输出：results/baseline_scores.csv
#       图表/fig04_k_evaluation.png
#       数据/features/cluster_euclidean.xlsx
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from config import (
    STANDARDIZED_DATA_PATH,
    CLUSTER_EUCLIDEAN_PATH,
    K_RANGE,
    RANDOM_SEED,
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
    print('05 基准聚类：K-means + 欧氏距离')
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
    # 第一部分：评估不同k值
    # ----------------------------------------------------------
    print('\n--- k值评估 ---')
    k_range = list(K_RANGE)
    sse_list = []
    silhouette_scores = []
    db_scores = []
    ch_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(X)

        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)

        sse_list.append(kmeans.inertia_)
        silhouette_scores.append(sil)
        db_scores.append(db)
        ch_scores.append(ch)

        print(f'  k={k}: SSE={kmeans.inertia_:.2f}, 轮廓系数={sil:.4f}, DB={db:.4f}, CH={ch:.2f}')

    # 保存评估结果CSV
    scores_df = pd.DataFrame({
        'k': k_range,
        'sse': sse_list,
        'silhouette': silhouette_scores,
        'davies_bouldin': db_scores,
        'calinski_harabasz': ch_scores
    })
    csv_path = os.path.join(RESULTS_FOLDER, 'baseline_scores.csv')
    scores_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\n  评估结果已保存：{csv_path}')

    # 绘制k值评估图
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('聚类数 k', fontsize=12)
    axes[0].set_ylabel('轮廓系数', fontsize=12)
    axes[0].set_title('轮廓系数曲线', fontsize=13)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, db_scores, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('聚类数 k', fontsize=12)
    axes[1].set_ylabel('DB指数', fontsize=12)
    axes[1].set_title('Davies-Bouldin指数', fontsize=13)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(k_range, sse_list, 'gs-', linewidth=2, markersize=8)
    axes[2].set_xlabel('聚类数 k', fontsize=12)
    axes[2].set_ylabel('SSE (Inertia)', fontsize=12)
    axes[2].set_title('SSE曲线', fontsize=13)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_FOLDER, 'fig04_k_evaluation.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  评估图已保存：{fig_path}')

    # ----------------------------------------------------------
    # 第二部分：选择最佳k并执行最终聚类
    # ----------------------------------------------------------
    best_k = k_range[np.argmax(silhouette_scores)]
    best_sil = max(silhouette_scores)
    best_db = db_scores[np.argmax(silhouette_scores)]

    print(f'\n--- 最终聚类 (k={best_k}) ---')
    print(f'  最佳k={best_k}: 轮廓系数={best_sil:.4f}, DB指数={best_db:.4f}')

    kmeans_final = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    labels_final = kmeans_final.fit_predict(X)

    # 将聚类结果添加到数据框
    df_result = df.copy()
    df_result['聚类标签'] = labels_final

    # 保存聚类结果
    df_result.to_excel(CLUSTER_EUCLIDEAN_PATH, index=False)
    print(f'  聚类结果已保存：{CLUSTER_EUCLIDEAN_PATH}')

    # 显示各聚类城市分布
    print('\n各聚类城市分布：')
    for cluster_id in range(best_k):
        cluster_cities = df_result[df_result['聚类标签'] == cluster_id]
        print(f'\n  聚类 {cluster_id} (共{len(cluster_cities)}个样本):')
        for year in YEARS:
            year_cities = cluster_cities[cluster_cities['年份'] == year]['城市'].tolist()
            if year_cities:
                print(f'    {year}年: {", ".join(year_cities)}')

    print('\n' + '=' * 50)
    print('基准聚类完成！')
    print(f'  最佳聚类数: k={best_k}')
    print(f'  轮廓系数: {best_sil:.4f}')
    print(f'  DB指数: {best_db:.4f}')


if __name__ == '__main__':
    main()
