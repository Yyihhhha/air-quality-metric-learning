# ============================================================
# config.py - 全局配置
# 面向空气质量数据相似性分析的度量学习与城市聚类
# ============================================================
import os

# ----------------------------------------------------------
# 路径配置
# ----------------------------------------------------------
# 项目根目录（脚本所在目录的父目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 数据文件夹路径
DATA_FOLDER = os.path.join(PROJECT_ROOT, "数据")

# 原始数据文件夹（存放年度Excel文件）
RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, "raw")

# 处理后的数据文件夹（存放每城市每日均值文件）
PROCESSED_DATA_FOLDER = os.path.join(DATA_FOLDER, "processed")

# 特征矩阵文件夹（存放标准化后的124×23特征矩阵）
FEATURES_FOLDER = os.path.join(DATA_FOLDER, "features")

# 图表输出文件夹
FIGURE_FOLDER = os.path.join(PROJECT_ROOT, "图表")

# 中间结果文件夹（CSV等）
RESULTS_FOLDER = os.path.join(PROJECT_ROOT, "results")

# 报告文件夹
REPORT_FOLDER = os.path.join(PROJECT_ROOT, "报告")

# 样例数据路径
SAMPLE_DATA_PATH = os.path.join(RAW_DATA_FOLDER, "sample.xlsx")

# 地图数据文件夹（存放GMT边界文件）
MAP_DATA_FOLDER = os.path.join(PROJECT_ROOT, "地图数据")

# ----------------------------------------------------------
# 数据参数
# ----------------------------------------------------------
# 31个省会城市+直辖市
CITIES = [
    '北京', '天津', '上海', '重庆',
    '石家庄', '太原', '呼和浩特',
    '沈阳', '长春', '哈尔滨',
    '南京', '杭州', '合肥', '福州', '南昌', '济南',
    '郑州', '武汉', '长沙', '广州', '南宁', '海口',
    '成都', '贵阳', '昆明', '拉萨',
    '西安', '兰州', '银川', '西宁', '乌鲁木齐'
]

# 需要保留的污染物指标
TYPES_NEEDED = ['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

# 有效小时数阈值（18小时）
HOUR_THRESHOLD = 18

# 年份和对应文件名
YEARS_FILES = {
    2016: '数据2016.xlsx',
    2017: '数据2017.xlsx',
    2018: '数据2018.xlsx',
    2019: '数据2019.xlsx',
}

YEARS = [2016, 2017, 2018, 2019]

# ----------------------------------------------------------
# 算法参数
# ----------------------------------------------------------
# K-means随机种子
RANDOM_SEED = 42

# 评估的聚类数范围
K_RANGE = range(2, 9)

# NCA参数
# NCA使用sklearn默认的L-BFGS优化器，学习率由内部自动调整，无需显式设置。
NCA_MAX_ITER = 1000

# LMNN参数
LMNN_K = 3
LMNN_LEARN_RATE = 1e-3
LMNN_MAX_ITER = 1000

# ----------------------------------------------------------
# 数据文件路径
# ----------------------------------------------------------
# 中间结果
COMBINED_RESULT_PATH = os.path.join(PROCESSED_DATA_FOLDER, "combined_result.xlsx")
STANDARDIZED_DATA_PATH = os.path.join(FEATURES_FOLDER, "standardized_data.xlsx")

# 聚类结果
CLUSTER_EUCLIDEAN_PATH = os.path.join(FEATURES_FOLDER, "cluster_euclidean.xlsx")
CLUSTER_NCA_PATH = os.path.join(FEATURES_FOLDER, "cluster_nca.xlsx")
CLUSTER_LMNN_PATH = os.path.join(FEATURES_FOLDER, "cluster_lmnn.xlsx")

# ----------------------------------------------------------
# 地理分区映射
# ----------------------------------------------------------
# 城市到地理分区的映射
CITY_TO_REGION = {
    # 华北 (0)
    '北京': 0, '天津': 0, '石家庄': 0, '太原': 0, '呼和浩特': 0,
    # 东北 (1)
    '沈阳': 1, '长春': 1, '哈尔滨': 1,
    # 华东 (2)
    '上海': 2, '南京': 2, '杭州': 2, '合肥': 2, '福州': 2, '南昌': 2, '济南': 2,
    # 中南 (3)
    '郑州': 3, '武汉': 3, '长沙': 3, '广州': 3, '南宁': 3, '海口': 3,
    # 西南 (4)
    '重庆': 4, '成都': 4, '贵阳': 4, '昆明': 4, '拉萨': 4,
    # 西北 (5)
    '西安': 5, '兰州': 5, '西宁': 5, '银川': 5, '乌鲁木齐': 5,
}

# 分区名称映射
REGION_NAMES = {
    0: '华北', 1: '东北', 2: '华东', 3: '中南', 4: '西南', 5: '西北'
}

# ----------------------------------------------------------
# 工具函数
# ----------------------------------------------------------
def ensure_dirs():
    """确保所有输出目录存在"""
    for folder in [PROCESSED_DATA_FOLDER, FEATURES_FOLDER, FIGURE_FOLDER, RESULTS_FOLDER]:
        os.makedirs(folder, exist_ok=True)
