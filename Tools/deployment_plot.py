import pandas as pd
import numpy as np
from Model import config
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import rasterio
from rasterio.windows import from_bounds

# --- 全局绘图设置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# === 关键：恢复您原始代码中的颜色定义 ===
# 对应关系: 0:Blue, 1:Orange(Yellowish), 2:Green, 3:Red, 4:Purple
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def get_color_for_type(cap_type):
    # 逻辑：Type 1 -> idx 0 (Blue)
    #       Type 2 -> idx 1 (Orange/Yellow)
    #       Type 3 -> idx 2 (Green)
    #       Type 4 -> idx 3 (Red)
    idx = max(0, int(cap_type) - 1)
    return color_palette[idx % len(color_palette)]

def calculate_real_terrain_polygon(nest_row, physical_model, num_points=72):
    center_lon = nest_row['lon']
    center_lat = nest_row['lat']
    center_z = nest_row['z_ground']
    
    polygon_points_lonlat = []
    mean_lat_rad = np.radians(center_lat)
    m_per_deg_lat = 111000
    m_per_deg_lon = 111000 * np.cos(mean_lat_rad)

    thetas = np.linspace(0, 360, num_points, endpoint=False)

    for theta in thetas:
        d_limit = physical_model.calculate_max_horizontal_distance_limit(center_lon, center_lat, center_z, theta)
        if d_limit > 0:
            theta_rad = np.radians(theta)
            dy_m = d_limit * np.cos(theta_rad) 
            dx_m = d_limit * np.sin(theta_rad) 
            d_lat = dy_m / m_per_deg_lat
            d_lon = dx_m / m_per_deg_lon
            polygon_points_lonlat.append([center_lon + d_lon, center_lat + d_lat])
            
    if len(polygon_points_lonlat) > 0:
        polygon_points_lonlat.append(polygon_points_lonlat[0])
        
    return np.array(polygon_points_lonlat)

def plot_results(physical_model_inst, phase1_results):
    selected_nests_df, assignments_df, objective_value = phase1_results
    if selected_nests_df is None: return

    print("\n--- Plotting Phase I Results (Color Fixed) ---")

    nest_locations = physical_model_inst.nest_locations.copy()
    turbines_df = physical_model_inst.wind_turbines_df.copy().reset_index(drop=True)

    turbines_df['id'] = turbines_df['id'].astype(int)
    nest_locations['id'] = nest_locations['id'].astype(int)
    if selected_nests_df is not None:
        selected_nests_df['nest_id'] = selected_nests_df['nest_id'].astype(int)

    nest_locations['is_selected'] = False
    nest_locations['capacity'] = 0 
    if selected_nests_df is not None:
        cap_map = dict(zip(selected_nests_df['nest_id'], selected_nests_df['capacity']))
        for index, row in nest_locations.iterrows():
            nid = int(row['id'])
            if nid in cap_map:
                nest_locations.at[index, 'is_selected'] = True
                nest_locations.at[index, 'capacity'] = cap_map[nid]

    turbines_df['assigned_nest_id'] = -1 
    if assignments_df is not None:
        assign_map = dict(zip(assignments_df['turbine_id'].astype(int), assignments_df['nest_id'].astype(int)))
        for idx in turbines_df.index:
            if idx in assign_map:
                turbines_df.at[idx, 'assigned_nest_id'] = assign_map[idx]

    # --- 投影 ---
    all_lons = pd.concat([nest_locations['lon'], turbines_df['lon']])
    all_lats = pd.concat([nest_locations['lat'], turbines_df['lat']])
    ref_lon, ref_lat = all_lons.min(), all_lats.min()
    m_per_deg_lat = 111000.0
    m_per_deg_lon = 111000.0 * np.cos(np.radians(ref_lat))

    def to_xy(lon, lat):
        x = (lon - ref_lon) * m_per_deg_lon
        y = (lat - ref_lat) * m_per_deg_lat
        return x, y

    nest_locations['x'], nest_locations['y'] = to_xy(nest_locations['lon'], nest_locations['lat'])
    turbines_df['x'], turbines_df['y'] = to_xy(turbines_df['lon'], turbines_df['lat'])

    # --- 计算多边形 ---
    selected_nests_only = nest_locations[nest_locations['is_selected']]
    nest_polygons_m = {} 
    print("Calculating terrain-aware coverage polygons...")

    all_x = [turbines_df['x'], nest_locations['x']] 
    all_y = [turbines_df['y'], nest_locations['y']]

    for idx, row in selected_nests_only.iterrows():
        nid = int(row['id'])
        poly_geo = calculate_real_terrain_polygon(row, physical_model_inst)
        if len(poly_geo) > 0:
            px, py = to_xy(poly_geo[:, 0], poly_geo[:, 1])
            nest_polygons_m[nid] = np.column_stack((px, py))
            all_x.append(pd.Series(px)); all_y.append(pd.Series(py))

    # --- ROI ---
    concat_x = pd.concat(all_x); concat_y = pd.concat(all_y)
    min_x, max_x = concat_x.min(), concat_x.max()
    min_y, max_y = concat_y.min(), concat_y.max()
    buffer_x = (max_x - min_x) * 0.05
    buffer_y = (max_y - min_y) * 0.05
    roi_x_min, roi_x_max = min_x - buffer_x, max_x + buffer_x
    roi_y_min, roi_y_max = min_y - buffer_y, max_y + buffer_y
    
    roi_lon_min = roi_x_min / m_per_deg_lon + ref_lon
    roi_lon_max = roi_x_max / m_per_deg_lon + ref_lon
    roi_lat_min = roi_y_min / m_per_deg_lat + ref_lat
    roi_lat_max = roi_y_max / m_per_deg_lat + ref_lat

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 8)) 

    # DEM
    try:
        with rasterio.open(config.DEM_FILE) as src:
            map_left, map_bottom, map_right, map_top = src.bounds
            valid_left = max(roi_lon_min, map_left); valid_right = min(roi_lon_max, map_right)
            valid_bottom = max(roi_lat_min, map_bottom); valid_top = min(roi_lat_max, map_top)
            
            if valid_left < valid_right and valid_bottom < valid_top:
                window = from_bounds(valid_left, valid_bottom, valid_right, valid_top, src.transform)
                elevation_data = src.read(1, window=window)
                elevation_data = np.where(elevation_data < -1000, np.nan, elevation_data)
                
                ex_x_min, ex_y_min = to_xy(valid_left, valid_bottom)
                ex_x_max, ex_y_max = to_xy(valid_right, valid_top)
                
                im = ax.imshow(elevation_data, cmap='terrain', 
                               extent=[ex_x_min, ex_x_max, ex_y_min, ex_y_max], 
                               alpha=0.5, origin='upper', 
                               vmin=physical_model_inst.min_elevation_dem, 
                               vmax=physical_model_inst.max_elevation_dem)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Elevation (m)')

                # Fix blank area
                roi_x_min, roi_x_max = ex_x_min, ex_x_max
                roi_y_min, roi_y_max = ex_y_min, ex_y_max
    except Exception as e:
        print(f"Could not plot DEM: {e}")

    # Elements
    unselected = nest_locations[~nest_locations['is_selected']]
    ax.scatter(unselected['x'], unselected['y'], c='lightgray', s=30, marker='o', edgecolors='gray', zorder=2)
    ax.scatter(turbines_df['x'], turbines_df['y'], c='gray', marker='x', s=40, zorder=3)

    for idx, row in selected_nests_only.iterrows():
        nid = int(row['id']); cap = int(row['capacity'])
        # 使用原来的函数获取颜色
        color = get_color_for_type(cap)
        
        if nid in nest_polygons_m:
            poly = nest_polygons_m[nid]
            ax.plot(poly[:, 0], poly[:, 1], color=color, linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

        my_turbines = turbines_df[turbines_df['assigned_nest_id'] == nid]
        for _, t_row in my_turbines.iterrows():
            ax.plot([row['x'], t_row['x']], [row['y'], t_row['y']], c=color, lw=0.8, alpha=0.4, zorder=2)

        ax.scatter(row['x'], row['y'], c=color, s=60, marker='s', edgecolors='k', zorder=5)
        ax.text(row['x'], row['y'] + (roi_y_max-roi_y_min)*0.02, f"N{nid}", ha='center', fontsize=8, fontweight='bold', zorder=6)

    # Wind Arrow
    Wx, Wy = config.WIND_SPEED
    W_mag = np.sqrt(Wx**2 + Wy**2)
    nav_angle = (90 - np.degrees(np.arctan2(Wy, Wx))) % 360
    
    width_map = roi_x_max - roi_x_min
    arrow_len = width_map * 0.08
    arrow_x = roi_x_min + width_map * 0.08
    arrow_y = roi_y_max - (roi_y_max-roi_y_min) * 0.08
    
    ax.arrow(arrow_x, arrow_y, (Wx/W_mag)*arrow_len, (Wy/W_mag)*arrow_len, 
             head_width=width_map*0.02, width=width_map*0.005, fc='blue', ec='blue', zorder=10)
    ax.text(arrow_x, arrow_y - width_map*0.05, f"Wind: {W_mag}m/s\nDir: {nav_angle:.0f}°", color='blue', ha='center', fontsize=8, fontweight='bold', zorder=10)

    # Decor
    ax.set_title(f"Optimization Result (Nests: {len(selected_nests_only)}, Cost: {objective_value:.2f})", fontsize=14)
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_xlim(roi_x_min, roi_x_max); ax.set_ylim(roi_y_min, roi_y_max)
    ax.set_aspect('equal')
    
    # Legend
    handles = [
        Line2D([0],[0], marker='x', color='gray', label='Turbine', ls='None'),
        Line2D([0],[0], marker='o', color='lightgray', label='Candidate', ls='None'),
        Line2D([0],[0], linestyle='--', color='gray', label='Max Energy Range')
    ]
    for c_val in sorted(selected_nests_only['capacity'].unique()):
        # Legend 也必须使用相同的颜色函数
        handles.append(Line2D([0],[0], marker='s', color=get_color_for_type(c_val), label=f'Nest Type {c_val}', ls='None'))
    
    ax.legend(handles=handles, loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.show()
