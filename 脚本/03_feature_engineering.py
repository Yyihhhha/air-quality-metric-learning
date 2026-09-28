# ============================================================
# 03_feature_engineering.py
# 功能：对合并后的特征数据进行Z-score标准化
# 输入：数据/processed/combined_result.xlsx
# 输出：数据/features/standardized_data.xlsx
# ============================================================
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import (
    COMBINED_RESULT_PATH,
    STANDARDIZED_DATA_PATH,
    ensure_dirs
)


def main():
    ensure_dirs()

    print('=' * 50)
    print('03 特征工程（标准化）')
    print('=' * 50)

    # 读取合并后的数据
    if not os.path.exists(COMBINED_RESULT_PATH):
        print(f'[错误] 文件不存在：{COMBINED_RESULT_PATH}')
        print('  请先运行 02_merge_data.py')
        exit(1)

    df = pd.read_excel(COMBINED_RESULT_PATH)

    # 需要标准化的特征列（排除城市和年份）
    feature_cols = [col for col in df.columns if col not in ['城市', '年份']]

    print(f'原始数据形状：{df.shape}')
    print(f'需要标准化的特征数：{len(feature_cols)}')
    print(f'特征列：{feature_cols}')

    # Z-score标准化
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

    # 保存标准化后的数据
    df_scaled.to_excel(STANDARDIZED_DATA_PATH, index=False)

    print(f'\n标准化完成！')
    print(f'  输出形状：{df_scaled.shape}')
    print(f'  保存到：{STANDARDIZED_DATA_PATH}')


if __name__ == '__main__':
    main()
