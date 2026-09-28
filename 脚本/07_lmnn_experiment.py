# ============================================================
# 07_lmnn_experiment.py
# 功能：LMNN度量学习 + K-means 聚类实验
# 输入：数据/features/standardized_data.xlsx
#       数据/features/cluster_euclidean.xlsx（用作伪标签来源）
# 输出：results/lmnn_scores.csv
#       图表/fig06_lmnn_comparison.png
#       数据/features/cluster_lmnn.xlsx
# 依赖：metric-learn>=0.7.0
# ============================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from config import (
    STANDARDIZED_DATA_PATH,
    CLUSTER_EUCLIDEAN_PATH,
    CLUSTER_LMNN_PATH,
    K_RANGE,
    RANDOM_SEED,
    LMNN_K,
    LMNN_LEARN_RATE,
    LMNN_MAX_ITER,
    YEARS,
    FIGURE_FOLDER,
    RESULTS_FOLDER,
    ensure_dirs
)

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入metric_learn
LMNN_AVAILABLE = False
try:
    from metric_learn import LMNN
    LMNN_AVAILABLE = True
    print('[OK] metric_learn库导入成功')
except ImportError:
    print('[警告] metric_learn库未安装，请运行: pip install metric-learn>=0.7.0')
    print('  将跳过LMNN实验')


def main():
    ensure_dirs()

    print('=' * 50)
    print('07 LMNN度量学习 + K-means 聚类')
    print('=' * 50)

    if not LMNN_AVAILABLE:
        print('\n[跳过] 请先安装 metric-learn：pip install metric-learn')
        return

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
    # LMNN训练
    # ----------------------------------------------------------
    print('\n--- 训练LMNN ---')
    try:
        lmnn = LMNN(
            k=LMNN_K,
            learn_rate=LMNN_LEARN_RATE,
            max_iter=LMNN_MAX_ITER,
            verbose=False
        )
        X_lmnn = lmnn.fit_transform(X, y_pseudo)
        print(f'LMNN训练完成！新空间维度：{X_lmnn.shape[1]}')
    except Exception as e:
        print(f'[错误] LMNN训练失败：{e}')
        print('  提示：可尝试 pip install metric-learn==0.7.0')
        return

    # LMNN变换矩阵分析
    print('\n--- LMNN变换矩阵分析 ---')
    if hasattr(lmnn, 'components_'):
        frobenius_norm = np.linalg.norm(lmnn.components_)
        identity_diff = np.linalg.norm(lmnn.components_ - np.eye(len(feature_cols)))
        print(f'  变换矩阵Frobenius范数: {frobenius_norm:.4f}')
        print(f'  与单位矩阵差异: {identity_diff:.4f}')
        if identity_diff < 1e-3:
            print('  [警告] LMNN变换矩阵接近单位矩阵，未学到有效变换')
    X_diff_norm = np.linalg.norm(X_lmnn - X) / (np.linalg.norm(X) + 1e-8)
    print(f'  变换前后数据相对差异: {X_diff_norm:.4f}')

    # ----------------------------------------------------------
    # 在LMNN空间中进行K-means聚类
    # ----------------------------------------------------------
    print('\n--- LMNN空间K-means聚类 ---')
    k_range = list(K_RANGE)
    lmnn_scores = []
    lmnn_db = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(X_lmnn)

        sil = silhouette_score(X_lmnn, labels)
        db = davies_bouldin_score(X_lmnn, labels)
        lmnn_scores.append(sil)
        lmnn_db.append(db)

        print(f'  k={k}: 轮廓系数={sil:.4f}, DB指数={db:.4f}')

    # 保存评估结果CSV
    scores_df = pd.DataFrame({
        'k': k_range,
        'silhouette': lmnn_scores,
        'davies_bouldin': lmnn_db
    })
    csv_path = os.path.join(RESULTS_FOLDER, 'lmnn_scores.csv')
    scores_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\n  评估结果已保存：{csv_path}')

    # ----------------------------------------------------------
    # 选择最佳k，执行最终聚类
    # ----------------------------------------------------------
    best_k_lmnn = k_range[np.argmax(lmnn_scores)]
    best_sil_lmnn = max(lmnn_scores)
    print(f'\n  最佳聚类数: k={best_k_lmnn}, 轮廓系数={best_sil_lmnn:.4f}')

    kmeans_lmnn_final = KMeans(n_clusters=best_k_lmnn, random_state=RANDOM_SEED, n_init=10)
    labels_lmnn_final = kmeans_lmnn_final.fit_predict(X_lmnn)

    # 保存LMNN聚类结果
    df_lmnn_result = df.copy()
    df_lmnn_result['聚类标签_LMNN'] = labels_lmnn_final
    df_lmnn_result.to_excel(CLUSTER_LMNN_PATH, index=False)
    print(f'  聚类结果已保存：{CLUSTER_LMNN_PATH}')

    # 显示各聚类城市分布
    print('\nLMNN各聚类城市分布：')
    for cluster_id in range(best_k_lmnn):
        cluster_cities = df_lmnn_result[df_lmnn_result['聚类标签_LMNN'] == cluster_id]
        print(f'\n  聚类 {cluster_id} (共{len(cluster_cities)}个样本):')
        for year in YEARS:
            year_cities = cluster_cities[cluster_cities['年份'] == year]['城市'].tolist()
            if year_cities:
                print(f'    {year}年: {", ".join(year_cities)}')

    # ----------------------------------------------------------
    # 对比图：LMNN vs 欧氏距离
    # ----------------------------------------------------------
    # 计算欧氏距离基准（原始空间）
    labels_euclidean_base = KMeans(n_clusters=best_k_lmnn, random_state=RANDOM_SEED, n_init=10).fit_predict(X)
    sil_euclidean = silhouette_score(X, labels_euclidean_base)
    db_euclidean = davies_bouldin_score(X, labels_euclidean_base)

    print(f'\n--- 对比 ---')
    print(f'  欧氏距离 (k={best_k_lmnn}): 轮廓系数={sil_euclidean:.4f}, DB指数={db_euclidean:.4f}')
    print(f'  LMNN     (k={best_k_lmnn}): 轮廓系数={best_sil_lmnn:.4f}, DB指数={lmnn_db[best_k_lmnn - 2]:.4f}')

    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(k_range, lmnn_scores, 'r^-', linewidth=2, markersize=8, label='LMNN')
    axes[0].axhline(y=sil_euclidean, color='b', linestyle='--', linewidth=2, label=f'欧氏距离 (k={best_k_lmnn})')
    axes[0].set_xlabel('聚类数 k', fontsize=12)
    axes[0].set_ylabel('轮廓系数', fontsize=12)
    axes[0].set_title('LMNN vs 欧氏距离 - 轮廓系数', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, lmnn_db, 'r^-', linewidth=2, markersize=8, label='LMNN')
    axes[1].axhline(y=db_euclidean, color='b', linestyle='--', linewidth=2, label=f'欧氏距离 (k={best_k_lmnn})')
    axes[1].set_xlabel('聚类数 k', fontsize=12)
    axes[1].set_ylabel('DB指数', fontsize=12)
    axes[1].set_title('LMNN vs 欧氏距离 - DB指数', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_FOLDER, 'fig06_lmnn_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  对比图已保存：{fig_path}')

    print('\n' + '=' * 50)
    print('LMNN实验完成！')


if __name__ == '__main__':
    main()
