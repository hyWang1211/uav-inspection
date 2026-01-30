#version2:设置空速和地速不超过u_max
import os
import numpy as np
import pandas as pd
import math
# --- 目录配置 ---
DATASET_DIR = 'dataset'
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 数据文件 ---
OSM_FILE = os.path.join(DATASET_DIR, 'subsets/turbines_200_clustered.csv')
DEM_FILE = os.path.join(DATASET_DIR, 'n41_e114_1arc_v3.tif') 

# --- 物理与工程参数 (Based on uav_inspection8.md) ---
UAV_MASS = 10.0      # kg
G = 9.8              # m/s^2
RHO = 1.225          # kg/m^3
AREA = 0.5           # m^2
U_TIP = 120.0        # m/s
D_0 = 0.3
SOLIDITY = 0.05
C_D0 = 0.01
KAPPA = 1.15

# !!! 核心修改：速度限制 !!!
# 既是最大允许地速，也是最大允许空速
U_MAX = 15.0         # m/s (约 54 km/h)
# U_MAX = 10.0         # m/s (约 54 km/h)

V_UP = 3.0           # m/s
V_DOWN = 3.0         # m/s
T_SERV = 60.0        # s
T_SWAP = 300.0       # s
DELTA_SAFE = 20.0    # m
E_BAT = 1000000.0    # J

WIND_SPEED = (3.0, 2.0) 

# --- 成本参数 ---
OMEGA_2 = 1      # 无人机成本
NEST_BASE_COSTS = {
    1: 2,       # 1架容量机巢成本
    2: 3.8,       # 2架容量机巢成本
    3: 5.4,       # 3架容量机巢成本
    4: 6.8        # 4架容量机巢成本
}

# OMEGA_2 = 1      # 无人机成本
# NEST_BASE_COSTS = {
#     1: 2,       # 1架容量机巢成本
#     2: 4,       # 2架容量机巢成本
#     3: 6,       # 3架容量机巢成本
#     4: 8        # 4架容量机巢成本
# }

ETA = 4              # 单机最大服务风机数
MAX_ROUTS = 4        

# --- 求解器配置 ---
OPTIMIZER = 'PULP_CBC_CMD' 
SOLVER_TIME_LIMIT = 30 

# --- 离散化参数 ---
NUM_THETA_POINTS = 360 
SOLVER_PRECISION = 1e-3 


# 动态修改修饰机巢候选点个数
DEFAULT_TURBINES = 100  # 设置一个默认值，防止第一次运行文件还没生成时报错

if os.path.exists(OSM_FILE):
    try:
        _df_temp = pd.read_csv(OSM_FILE)
        N_TURBINES = len(_df_temp)
        print(f"检测到数据文件，风机数量设置为: {N_TURBINES}")
    except Exception as e:
        print(f"读取数据文件失败，使用默认值: {e}")
        N_TURBINES = DEFAULT_TURBINES
else:
    # 文件不存在（可能是第一次运行，还没生成）
    N_TURBINES = DEFAULT_TURBINES
    print(f"数据文件未找到，使用默认风机数量: {N_TURBINES}")

N_NESTS = math.ceil(N_TURBINES / 10.0)
# 确保至少有1个机巢
if N_NESTS < 1:
    N_NESTS = 1

# --- 示例数据规模 ---
# N_NESTS = 80
# N_TURBINES = 20

NEST_CAPACITIES = [1, 2, 3, 4] 

# --- 模拟数据生成函数 ---
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

def generate_simulated_data(n_nests, n_turbines, demo_file=DEM_FILE, osm_file=OSM_FILE):
    print("生成模拟数据...")
    if not os.path.exists(demo_file) or os.path.getsize(demo_file) == 0:
        height, width = 100, 100
        left, bottom, right, top = 114.5, 41.0, 115.0, 41.3
        x = np.linspace(left, right, width); y = np.linspace(bottom, top, height)
        X, Y = np.meshgrid(x, y)
        Z = 200 * np.sin(X * 5 + Y * 7) + 300
        transform = from_bounds(left, bottom, right, top, height, width)
        with rasterio.open(demo_file, 'w', driver='GTiff', height=height, width=width, count=1, dtype=Z.dtype, crs='EPSG:4326', transform=transform) as ds:
            ds.write(Z.astype(rasterio.float32), 1)

    if not os.path.exists(osm_file) or os.path.getsize(osm_file) == 0:
        with rasterio.open(demo_file) as src:
            b = src.bounds
            lats = np.random.uniform(b.bottom, b.top, n_turbines)
            lons = np.random.uniform(b.left, b.right, n_turbines)
        pd.DataFrame({'id': range(n_turbines), 'lat': lats, 'lon': lons}).to_csv(osm_file, index=False)
    return True