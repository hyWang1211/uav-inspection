import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import numpy as np

# ==========================================
# 全局设置：Times New Roman 字体
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False 

# ================= 配置 =================
# dataset_dir = 'dataset'
# osm_file = os.path.join(dataset_dir, 'zhangbei_turbines_osm.csv')

# 对于划分的数据集
dataset_dir = 'dataset'
osm_file = os.path.join(dataset_dir, 'subsets/turbines_200_clustered.csv')
DOWNSAMPLE_FACTOR = 2
# =======================================

# 1. 文件准备
tif_files = glob.glob(os.path.join(dataset_dir, '*.tif'))
if not tif_files:
    print("Error: No .tif file found.")
    exit()
tif_path = tif_files[0]
df = pd.read_csv(osm_file)

with rasterio.open(tif_path) as src:
    print(f"正在处理数据...")
    bounds = src.bounds
    # 这里的 extent 是 [左, 右, 下, 上]
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    
    # 读取全分辨率数据
    elevation_full = src.read(1)
    elevation_full = np.where(elevation_full < -1000, np.nan, elevation_full)
    
    # 筛选风机
    mask = (
        (df['lon'] >= bounds.left) & (df['lon'] <= bounds.right) & 
        (df['lat'] >= bounds.bottom) & (df['lat'] <= bounds.top)
    )
    df_plot = df[mask].copy()
    
    # 获取风机高度
    turbine_z = []
    coords = [(row['lon'], row['lat']) for _, row in df_plot.iterrows()]
    for val in src.sample(coords):
        z = val[0]
        if z < -1000: z = np.nanmin(elevation_full)
        turbine_z.append(z)
    df_plot['z_ground'] = turbine_z

    # ==========================================
    #             图 1: 2D 平面图
    # ==========================================
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    
    # 画地形 (2D imshow 自动处理方向，origin='upper' 是默认行为)
    im = ax1.imshow(elevation_full, cmap='terrain', extent=extent, alpha=0.8)
    plt.colorbar(im, ax=ax1, label='Elevation (m)')
    
    # 画风机 (红色)
    ax1.scatter(df_plot['lon'], df_plot['lat'], c='red', s=15, marker='^', edgecolors='white', linewidth=0.3, label='Turbines')
    
    ax1.set_title("2D Map")
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    # ax1.legend(loc='lower right')

    # ==========================================
    #             图 2: 3D 透视图 (已修正)
    # ==========================================
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111, projection='3d')
    
    # 1. 降采样
    new_h = src.height // DOWNSAMPLE_FACTOR
    new_w = src.width // DOWNSAMPLE_FACTOR
    elevation_down = src.read(1, out_shape=(new_h, new_w), resampling=rasterio.enums.Resampling.bilinear)
    elevation_down = np.where(elevation_down < -1000, np.nan, elevation_down)
    
    # 2. 生成网格
    x = np.linspace(bounds.left, bounds.right, new_w)
    y = np.linspace(bounds.bottom, bounds.top, new_h)
    X, Y = np.meshgrid(x, y)
    
    # --- 核心修正：翻转矩阵 ---
    # 因为 meshgrid 的 Y 是从下到上 (low->high)
    # 而 rasterio 读出的矩阵第0行是地图最上面 (High Lat)
    # 所以必须把矩阵上下颠倒，才能和 Y 轴对应上！
    elevation_down_fixed = np.flipud(elevation_down)
    
    # 3. 画地形
    surf = ax2.plot_surface(X, Y, elevation_down_fixed, cmap='terrain', alpha=0.6, linewidth=0, antialiased=False)
    
    # 4. 画风机
    z_range = np.nanmax(elevation_full) - np.nanmin(elevation_full)
    visual_height = z_range * 0.1
    z_display = df_plot['z_ground'] + visual_height
    
    ax2.scatter(df_plot['lon'], df_plot['lat'], z_display, 
                c='red', marker='^', s=40, edgecolors='k', linewidth=0.5, depthshade=False,
                label='Turbines')
    
    ax2.set_title("3D Map")
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    
    # # 视角设置：稍微俯视一点，方便看分布
    ax2.set_box_aspect((1, 1, 0.4))
    # ax2.view_init(elev=60, azim=-90) # 俯视60度，正北朝上

plt.show()