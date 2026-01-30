import overpy
import pandas as pd

# 1. 定义张北附近的大致中心点和搜索半径
# 张北县大致坐标 (41.15, 114.70), 搜索半径 20000米 (20公里)
lat, lon = 41.15, 114.70
radius = 20000

api = overpy.Overpass()

# 2. 构建查询语句 (Overpass QL)
# 意思：在 (lat, lon) 周围 radius 米范围内，寻找所有 "generator:source=wind" 的节点
query = f"""
[out:json];
(
  node["generator:source"="wind"](around:{radius},{lat},{lon});
  way["generator:source"="wind"](around:{radius},{lat},{lon});
  relation["generator:source"="wind"](around:{radius},{lat},{lon});
);
out center;
"""

print("正在向 OpenStreetMap 请求数据 (可能需要十几秒)...")
result = api.query(query)

turbines = []
for node in result.nodes:
    turbines.append({'id': node.id, 'lat': float(node.lat), 'lon': float(node.lon)})

# 如果有些风机是画成“区域(way)”的，取它们的中心点
for way in result.ways:
    turbines.append({'id': way.id, 'lat': float(way.center_lat), 'lon': float(way.center_lon)})

print(f"成功找到 {len(turbines)} 台风机！")

# 3. 保存为 CSV
if len(turbines) > 0:
    df = pd.DataFrame(turbines)
    df.to_csv('dataset/zhangbei_turbines_osm.csv', index=False)
    print("坐标已保存至 dataset/zhangbei_turbines_osm.csv")
    print(df.head())
else:
    print("该区域未找到公开的风机数据，请尝试扩大半径或更换坐标。")