# ============================================================
# 01_data_preprocessing.py
# 功能：读取原始Excel数据，筛选污染物指标，计算日均值
# 输入：数据/raw/数据{年份}.xlsx
# 输出：数据/processed/{城市}_{年份}_daily.xlsx
# ============================================================
import os
import pandas as pd
from config import (
    RAW_DATA_FOLDER,
    PROCESSED_DATA_FOLDER,
    CITIES,
    TYPES_NEEDED,
    HOUR_THRESHOLD,
    YEARS_FILES,
    ensure_dirs
)


def main():
    ensure_dirs()

    print('=' * 50)
    print('01 数据预处理')
    print('=' * 50)

    total_processed = 0

    for year, filename in YEARS_FILES.items():
        file_path = os.path.join(RAW_DATA_FOLDER, filename)

        if not os.path.exists(file_path):
            print(f'\n[跳过] 文件不存在：{filename}')
            continue

        print(f'\n正在处理 {year} 年数据：{filename}')

        try:
            df = pd.read_excel(file_path)
            print(f'  原始数据行数：{len(df)}')

            # 删除第一列（Source.Name），如果存在
            if 'Source.Name' in df.columns:
                df = df.drop(columns=['Source.Name'])
                print(f'  已删除 Source.Name 列')

            # 筛选需要的污染物指标
            df = df[df['type'].isin(TYPES_NEEDED)]
            print(f'  筛选指标后行数：{len(df)}')

            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')

            # 处理每个城市
            year_processed = 0
            for city in CITIES:
                try:
                    if city not in df.columns:
                        continue

                    # 提取该城市的数据
                    df_city = df[['date', 'type', city]].copy()
                    df_city.columns = ['date', 'type', 'value']

                    # 删除空值
                    df_city = df_city.dropna(subset=['value'])

                    if len(df_city) == 0:
                        continue

                    # 计算每天每种污染物有多少小时有数据
                    hourly_count = df_city.groupby(['date', 'type']).size().reset_index(name='hour_count')

                    # 筛选有效小时数 >= 阈值的记录
                    valid_dates = hourly_count[hourly_count['hour_count'] >= HOUR_THRESHOLD]

                    # 只保留有效数据
                    df_valid = df_city.merge(valid_dates[['date', 'type']], on=['date', 'type'])

                    if len(df_valid) == 0:
                        continue

                    # 计算日均值（按日期和污染物类型分组取均值，然后展开为宽表）
                    daily_avg = df_valid.groupby(['date', 'type'])['value'].mean().unstack()

                    # 保存
                    output_file = os.path.join(PROCESSED_DATA_FOLDER, f'{city}_{year}_daily.xlsx')
                    daily_avg.to_excel(output_file)
                    year_processed += 1

                except Exception as e:
                    print(f'  [错误] {city} 处理出错：{e}')

            print(f'  [完成] {year}年：成功处理 {year_processed}/{len(CITIES)} 个城市')
            total_processed += year_processed

        except Exception as e:
            print(f'  [错误] 读取文件出错：{e}')

    print('\n' + '=' * 50)
    print(f'数据预处理完成！')
    print(f'  共处理 {total_processed} 个城市-年份文件')
    print(f'  输出文件夹：{PROCESSED_DATA_FOLDER}')


if __name__ == '__main__':
    main()
