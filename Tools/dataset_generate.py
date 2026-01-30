import pandas as pd
import numpy as np
import os

# ================= 配置 =================
INPUT_FILE = 'dataset/zhangbei_turbines_osm.csv'
OUTPUT_DIR = 'dataset/subsets'
SIZES = [20, 50, 100, 200, 400, 600, 1000]
# =======================================

def create_clustered_subsets():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件 {INPUT_FILE}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"正在读取: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    total_count = len(df)
    
    # 1. 计算所有风机的“几何中心” (重心)
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    print(f"风场几何中心坐标: ({center_lat:.6f}, {center_lon:.6f})")

    # 2. 计算每个风机到中心的距离 (用于排序)
    # 这里用简单的欧氏距离近似即可，目的是为了排序，不需要精确的米
    # (lat - center_lat)^2 + (lon - center_lon)^2
    # 注意：为了更精确一点，经度差要乘以 cos(lat) 的修正系数，但在小范围内不乘也行
    # 这里我们加上修正系数让它更圆一点
    mean_lat_rad = np.radians(center_lat)
    cos_lat = np.cos(mean_lat_rad)
    
    df['dist_to_center'] = np.sqrt(
        (df['lat'] - center_lat)**2 + 
        ((df['lon'] - center_lon) * cos_lat)**2
    )

    # 3. 按照距离从小到大排序
    # 这样排在最前面的，就是离中心最近的风机
    df_sorted = df.sort_values(by='dist_to_center').reset_index(drop=True)

    # 4. 截取并保存
    for size in SIZES:
        if size > total_count:
            print(f"跳过 {size} (超过原始数量)")
            size = total_count
            # continue
            
        # 取前 size 行 (也就是离中心最近的 size 个)
        subset_df = df_sorted.head(size).copy()
        
        # 删除辅助计算的 distance 列，保持文件干净
        subset_df = subset_df.drop(columns=['dist_to_center'])
        
        output_filename = f'turbines_{size}_clustered.csv'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        subset_df.to_csv(output_path, index=False)
        
        # 计算一下这个子集的边界范围，让你心里有数
        lat_span = subset_df['lat'].max() - subset_df['lat'].min()
        lon_span = subset_df['lon'].max() - subset_df['lon'].min()
        # 简单估算公里数 (1度 ≈ 111km)
        km_span_lat = lat_span * 111
        km_span_lon = lon_span * 111 * cos_lat
        
        print(f"--> 已生成: {output_filename}")
        print(f"    覆盖范围: 南北约 {km_span_lat:.2f} km, 东西约 {km_span_lon:.2f} km")

    print("\n所有聚集型子集生成完毕！")

if __name__ == "__main__":
    create_clustered_subsets()