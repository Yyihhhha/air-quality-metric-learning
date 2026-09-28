# ============================================================
# 02_merge_data.py
# 功能：读取每城市每年的日数据，提取统计特征，合并为完整数据集
# 输入：数据/processed/{城市}_{年份}_daily.xlsx
# 输出：数据/processed/combined_result.xlsx（124行×23列+城市年份标识）
# ============================================================
import os
import numpy as np
import pandas as pd
from config import (
    PROCESSED_DATA_FOLDER,
    COMBINED_RESULT_PATH,
    CITIES,
    YEARS,
    ensure_dirs
)

# 所有污染物（统一提取特征）
POLLUTANTS = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'AQI']


def main():
    ensure_dirs()

    print('=' * 50)
    print('02 数据合并与特征提取')
    print('=' * 50)

    # 检查processed_data文件夹是否存在
    if not os.path.exists(PROCESSED_DATA_FOLDER):
        print(f'[错误] 文件夹不存在：{PROCESSED_DATA_FOLDER}')
        print('  请先运行 01_data_preprocessing.py')
        exit(1)

    all_data = []

    for city in CITIES:
        for year in YEARS:
            filename = f'{city}_{year}_daily.xlsx'
            filepath = os.path.join(PROCESSED_DATA_FOLDER, filename)

            if not os.path.exists(filepath):
                print(f'  [跳过] 文件不存在：{filename}')
                continue

            try:
                df = pd.read_excel(filepath)

                # 确保date列是日期格式
                df['date'] = pd.to_datetime(df['date'])

                # 构建特征字典
                features = {
                    '城市': city,
                    '年份': year,
                }

                # 为每种污染物提取统计特征
                for pollutant in POLLUTANTS:
                    if pollutant in df.columns:
                        values = df[pollutant].dropna()
                        if len(values) > 0:
                            features[f'{pollutant}_avg'] = values.mean()
                            features[f'{pollutant}_std'] = values.std()
                            features[f'{pollutant}_max'] = values.max()
                        else:
                            features[f'{pollutant}_avg'] = np.nan
                            features[f'{pollutant}_std'] = np.nan
                            features[f'{pollutant}_max'] = np.nan

                # 优良天数比例（AQI <= 100）
                if 'AQI' in df.columns:
                    good_days = (df['AQI'] <= 100).sum()
                    total_days = len(df)
                    features['good_days_pct'] = (good_days / total_days) * 100 if total_days > 0 else 0
                else:
                    features['good_days_pct'] = np.nan

                # 重污染天数比例（AQI > 200）
                if 'AQI' in df.columns:
                    heavy_days = (df['AQI'] > 200).sum()
                    total_days = len(df)
                    features['heavy_pollution_pct'] = (heavy_days / total_days) * 100 if total_days > 0 else 0
                else:
                    features['heavy_pollution_pct'] = np.nan

                all_data.append(features)

            except Exception as e:
                print(f'  [错误] {city} {year}年处理出错：{e}')

    # 转换为DataFrame
    result_df = pd.DataFrame(all_data)

    # 删除包含NaN的行
    before_drop = len(result_df)
    result_df = result_df.dropna()
    after_drop = len(result_df)
    if before_drop > after_drop:
        print(f'\n  [警告] 删除了 {before_drop - after_drop} 行包含缺失值的数据')

    # 保存
    result_df.to_excel(COMBINED_RESULT_PATH, index=False)

    print('\n' + '=' * 50)
    print(f'数据合并完成！')
    print(f'  共 {len(result_df)} 行数据（应为 {len(CITIES) * len(YEARS)} 行）')
    print(f'  特征数：{len(result_df.columns) - 2}（不含城市、年份）')
    print(f'  保存到：{COMBINED_RESULT_PATH}')


if __name__ == '__main__':
    main()
