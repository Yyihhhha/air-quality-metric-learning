# 面向空气质量数据相似性分析的度量学习与城市聚类

## 1. 项目简介

本项目基于中国31个省会城市及直辖市2016-2019年的逐小时空气质量数据，构建城市年度空气质量特征矩阵（124样本 x 23特征），引入度量学习（Metric Learning）方法优化城市间的相似度度量，提升K-means聚类效果。

**核心思路：** 传统欧氏距离无法充分反映各污染物维度对城市空气质量分类的不同贡献。通过NCA（Neighborhood Components Analysis）和LMNN（Large Margin Nearest Neighbor）度量学习方法，学习一个最优线性变换矩阵，使聚类效果显著提升。

## 2. 主要结论

| 指标 | 欧氏距离 + K-means | NCA + K-means | LMNN + K-means |
|------|-------------------|---------------|----------------|
| 轮廓系数 | 0.3409 | **0.4506** | **0.4463** |
| DB指数 | 1.1609 | **0.8292** | 0.8629 |
| 训练时间(秒) | 0.406 | **0.036** | 0.400 |

- NCA度量学习使轮廓系数从0.3409提升到0.4506，**提升约32%**
- DB指数从1.1609降低到0.8292，**降低约29%**
- 三种方法均一致将城市分为"清洁城市"（南方为主）和"重污染城市"（北方为主）两类
- 重污染带呈条带状集聚，与冬季燃煤采暖区高度吻合

## 3. 目录结构

```
air-quality-metric-learning/
├── 数据/
│   ├── raw/                   # 原始Excel数据（年度小时数据）
│   │   └── sample.xlsx        # 样例数据（2天，可复现流程）
│   ├── processed/             # 清洗后的日均数据（124文件+合并结果）
│   └── features/              # 标准化特征矩阵与聚类结果
├── 脚本/
│   ├── config.py              # 全局配置（路径、参数）
│   ├── 01_data_preprocessing.py      # 数据预处理
│   ├── 02_merge_data.py              # 特征合并
│   ├── 03_feature_engineering.py     # Z-score标准化
│   ├── 04_eda_analysis.py            # 探索性数据分析
│   ├── 05_baseline_kmeans.py         # 基准K-means聚类
│   ├── 06_nca_experiment.py          # NCA度量学习实验
│   ├── 07_lmnn_experiment.py         # LMNN度量学习实验
│   ├── 08_evaluation.py              # 综合评估
│   └── 09_visualization.py           # 高级可视化（t-SNE+地图）
├── 图表/                      # 10张分析图表
├── results/                   # 5个CSV中间结果
├── 报告/                      # 分析报告.md
├── 地图数据/                  # GMT边界文件（国界线、省界线）
├── README.md
├── requirements.txt
└── .gitignore
```

## 4. 运行方式

### 环境准备

```bash
git clone <repo-url>
cd air-quality-metric-learning

# 创建虚拟环境（推荐）
conda create -n aqi python=3.13
conda activate aqi

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

由于原始数据文件较大（每年约230MB），不纳入版本管理。需手动将4个年度Excel文件放入 `数据/raw/` 目录：

```
数据/raw/
├── 数据2016.xlsx
├── 数据2017.xlsx
├── 数据2018.xlsx
└── 数据2019.xlsx
```

仓库中提供 `sample.xlsx`（2天样例数据）供快速测试流程。

### 按顺序执行脚本

```bash
cd 脚本/

python 01_data_preprocessing.py       # 数据预处理（小时→日均，约15分钟）
python 02_merge_data.py              # 特征合并（124样本×23特征）
python 03_feature_engineering.py     # Z-score标准化
python 04_eda_analysis.py            # 探索性分析（热力图、季节对比）
python 05_baseline_kmeans.py         # 基准K-means聚类
python 06_nca_experiment.py          # NCA度量学习实验
python 07_lmnn_experiment.py         # LMNN度量学习实验
python 08_evaluation.py              # 综合评估（指标对比+训练时间）
python 09_visualization.py           # 高级可视化（t-SNE+中国地图）
```

## 5. 技术栈

| 类别 | 工具/库 | 版本 | 用途 |
|------|---------|------|------|
| 运行环境 | Python | 3.13 | 主语言 |
| 数值计算 | NumPy | >=1.23 | 矩阵运算 |
| 数据处理 | Pandas | >=1.5 | Excel读取、特征计算 |
| 机器学习 | scikit-learn | >=1.2 | K-means、NCA、标准化、t-SNE |
| 度量学习 | metric-learn | >=0.7.0 | LMNN实现 |
| 可视化 | Matplotlib | >=3.6 | 图表绘制 |
| 统计可视化 | Seaborn | >=0.12 | 热力图、箱线图 |
| 地理信息 | Cartopy | >=0.21 | 中国地图聚类分布 |
| Excel读写 | openpyxl | >=3.0 | .xlsx文件处理 |

## 6. 数据来源

本项目使用中国环境监测总站发布的**全国城市空气质量实时监测数据**。

- **时间范围：** 2016年1月 - 2019年12月
- **数据粒度：** 逐小时（24小时/天）
- **城市覆盖：** 31个省会城市及直辖市
- **监测指标：** AQI、PM2.5、PM10、SO2、NO2、CO、O3
- **数据格式：** 每年一个Excel文件，约12.8万行（小时×日期×污染物类型）

由于原始数据文件总量约920MB，超出GitHub单文件限制（100MB），本仓库仅保留 `sample.xlsx` 作为格式参考。完整数据可通过中国环境监测总站官网或相关公开数据平台获取。

## 7. 参数设置

| 参数 | 值 | 说明 |
|------|-----|------|
| K-means 随机种子 | **42** | 保证可复现性 |
| 聚类数范围 | k = [2, 8] | 遍历确定最优k |
| 有效小时阈值 | >= 18 小时/天 | 日均值计算筛选条件 |
| NCA max_iter | 1000 | sklearn内部L-BFGS优化 |
| NCA 学习率 | 自动调整 | sklearn默认L-BFGS优化器自动调整，无需显式设置 |
| **LMNN K** | **3** | 每个类别的最近邻数 |
| LMNN learn_rate | 1e-3 | LMNN梯度下降学习率 |
| LMNN max_iter | 1000 | 最大迭代次数 |

所有参数集中配置于 `脚本/config.py`。

## 8. 特征工程说明

最终输入聚类模型的 **124 x 23** 特征矩阵构成：

| 特征组 | 维度 | 说明 |
|--------|------|------|
| 7种污染物年均值 | 7 | 各污染物年度平均水平 |
| 7种污染物标准差 | 7 | 各污染物年波动性 |
| 7种污染物最大值 | 7 | 各污染物的极端水平 |
| 优良天数比例 | 1 | AQI<=100天数占比 |
| 重污染天数比例 | 1 | AQI>200天数占比 |

124样本 = 31城市 x 4年份，每行为一个"城市-年"观测。

## 9. 完整图表清单

| 序号 | 文件 | 说明 |
|------|------|------|
| 1 | fig01_correlation_heatmap.png | 六项污染物相关系数热力图 |
| 2 | fig02_seasonal_comparison.png | 冬夏两季相关系数对比柱状图 |
| 3 | fig03_pm25_boxplot.png | PM2.5浓度季节分布箱线图 |
| 4 | fig04_k_evaluation.png | 欧氏距离k值评估（SSE/轮廓/DB三曲线） |
| 5 | fig05_nca_comparison.png | NCA vs 欧氏距离对比 |
| 6 | fig06_lmnn_comparison.png | LMNN vs 欧氏距离对比 |
| 7 | fig07_three_methods_compare.png | 三种方法综合对比 |
| 8 | fig08_training_time.png | 三种方法训练时间对比柱状图 |
| 9 | fig09_tsne_comparison.png | t-SNE三空间降维可视化 |
| 10 | fig10_cluster_map.png | 中国城市聚类空间分布地图 |

## License

本项目为独立数据分析项目，仅供技术展示与学习交流用途。
