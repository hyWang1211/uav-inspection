import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from Model import config 

# 尝试导入 KMeans，如果没有则使用备选方案
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class PhysicalModel:
    def __init__(self, dem_path, wind_speed):
        self.dem_path = dem_path
        self.wind_speed_vector = np.array(wind_speed) # (Wx, Wy) m/s
        
        self.dem_src = None
        self.load_dem()
        self.wind_turbines_df = self.load_wind_turbines()
        self.load_nest_locations()

        self._flight_time_cache = {}
        self._max_distance_limit_cache = {}
        
        # --- 初始化空气动力学常量 ---
        self._init_aerodynamics()

    def _init_aerodynamics(self):
        """预计算与状态无关的功率参数"""
        m = config.UAV_MASS
        g = config.G
        rho = config.RHO
        A = config.AREA
        sigma = config.SOLIDITY
        U_tip = config.U_TIP
        Cd0 = config.C_D0
        kappa = config.KAPPA
        
        # 悬停参数
        self.v0 = np.sqrt((m * g) / (2 * rho * A))
        
        term_induced = kappa * (m * g)**1.5 / np.sqrt(2 * rho * A)
        term_profile = (rho * sigma * A * U_tip**3 * Cd0) / 8.0
        self.P_hover = term_induced + term_profile
        
        # 垂直飞行参数近似
        self.P_climb_approx = self.P_hover + m * g * config.V_UP
        self.P_desc_approx = self.P_hover - m * g * config.V_DOWN 
        if self.P_desc_approx < 0: self.P_desc_approx = 100.0

        # 巡航功率计算系数
        self.const_profile = (rho * sigma * A * U_tip**3 * Cd0) / 8.0
        self.const_parasitic = 0.5 * config.D_0 * rho * A
        self.inv_U_tip_sq = 1.0 / (U_tip**2)

    def calculate_cruise_power(self, V_air_mag):
        """计算水平巡航功率 P_cruise"""
        term1_inner = np.sqrt(1 + (V_air_mag**4)/(4 * self.v0**4)) - (V_air_mag**2)/(2 * self.v0**2)
        term1_inner = np.maximum(term1_inner, 0.0) 
        P_induced = self.P_hover * np.sqrt(term1_inner)
        
        P_profile = self.const_profile * (1 + 3 * (V_air_mag**2) * self.inv_U_tip_sq)
        P_parasitic = self.const_parasitic * (V_air_mag**3)
        
        return P_induced + P_profile + P_parasitic

    def calculate_dynamic_flight_parameters(self, theta_deg):
        """
        【核心修改】根据最大速度限制 U_MAX 和风速，动态计算实际地速和功率。
        策略：
        1. 尝试保持地速 Vg = U_MAX。
        2. 如果这导致空速 V_air > U_MAX (逆风)，则限制 V_air = U_MAX，反推 Vg。
        
        :return: (Vg_actual, P_cruise_actual)
        """
        theta_rad = np.radians(theta_deg)
        Wx, Wy = self.wind_speed_vector
        U_max = config.U_MAX
        
        # 航向单位向量 (East, North)
        d_vec = np.array([np.sin(theta_rad), np.cos(theta_rad)])
        
        # --- 策略 1: 尝试最大地速 ---
        Vg_try = U_max
        V_ground_vec_try = Vg_try * d_vec
        V_air_vec_try = V_ground_vec_try - self.wind_speed_vector
        V_air_mag_try = np.linalg.norm(V_air_vec_try)
        
        if V_air_mag_try <= U_max:
            # 顺风或侧风，不需要满油门就能达到最大地速
            # 限制生效的是地速 (安全限制)
            return Vg_try, self.calculate_cruise_power(V_air_mag_try)
        else:
            # --- 策略 2: 逆风，限制最大空速 ---
            # 已知 V_air 的模长为 U_max
            # V_ground = k * d_vec
            # V_air = k * d_vec - W
            # || k * d - W ||^2 = U_max^2
            # k^2 - 2k(d . W) + |W|^2 - U_max^2 = 0
            
            W_dot_d = np.dot(self.wind_speed_vector, d_vec)
            W_sq = np.dot(self.wind_speed_vector, self.wind_speed_vector)
            
            # 解一元二次方程 ax^2 + bx + c = 0
            a = 1.0
            b = -2 * W_dot_d
            c = W_sq - U_max**2
            
            delta = b**2 - 4*a*c
            
            if delta < 0:
                # 理论上只要 U_max > WindSpeed，delta 恒大于 0
                # 如果风速大过飞机极速，飞不起来，返回 0 速度
                return 1e-6, 1e9 # 极慢速度，极大功率(惩罚)
                
            # 取正根 (地速必须为正)
            Vg_actual = (-b + np.sqrt(delta)) / (2*a)
            
            # 此时空速模长就是 U_max
            return Vg_actual, self.calculate_cruise_power(U_max)

    def load_dem(self):
        try:
            self.dem_src = rasterio.open(self.dem_path)
            self.elevation_full = self.dem_src.read(1)
            self.elevation_full = np.where(self.elevation_full < -1000, np.nan, self.elevation_full)
            self.bounds = self.dem_src.bounds
            self.transform = self.dem_src.transform
            self.width = self.dem_src.width
            self.height = self.dem_src.height
            self.nodata_val = self.dem_src.nodata if self.dem_src.nodata is not None else -9999
            
            valid_elevations = self.elevation_full[~np.isnan(self.elevation_full)]
            if valid_elevations.size > 0:
                self.min_elevation_dem = np.nanmin(valid_elevations)
                self.max_elevation_dem = np.nanmax(valid_elevations)
            else:
                self.min_elevation_dem = 0; self.max_elevation_dem = 0
        except Exception as e:
            print(f"Error opening DEM: {e}")
            raise

    def __del__(self):
        if self.dem_src:
            try: self.dem_src.close()
            except: pass

    def load_nest_locations(self):
        """使用 K-Means 聚类生成候选点"""
        n_nests = config.N_NESTS
        if not self.dem_src:
            self.nest_locations = pd.DataFrame(columns=['id', 'lon', 'lat', 'z_ground'])
            return

        if self.wind_turbines_df.empty:
            # 回退逻辑
            b = self.bounds
            lons = np.random.uniform(b.left, b.right, n_nests)
            lats = np.random.uniform(b.bottom, b.top, n_nests)
        else:
            print(f"Generating {n_nests} candidate sites using Smart Strategy...")
            turbine_coords = self.wind_turbines_df[['lon', 'lat']].values
            n_turbines = len(turbine_coords)
            
            nest_lons = []
            nest_lats = []

            if HAS_SKLEARN and n_nests < n_turbines:
                print(f"Using K-Means clustering...")
                kmeans = KMeans(n_clusters=n_nests, random_state=42, n_init=10)
                kmeans.fit(turbine_coords)
                centers = kmeans.cluster_centers_
                nest_lons = centers[:, 0]
                nest_lats = centers[:, 1]
            else:
                print("Using Hybrid Strategy...")
                jitter = 0.0005
                if n_nests < n_turbines:
                    indices = np.random.choice(n_turbines, n_nests, replace=False)
                    selected = turbine_coords[indices]
                else:
                    selected = turbine_coords
                
                for coord in selected:
                    nest_lons.append(coord[0] + np.random.uniform(-jitter, jitter))
                    nest_lats.append(coord[1] + np.random.uniform(-jitter, jitter))
                
                remaining = n_nests - len(nest_lons)
                if remaining > 0:
                    t_min_lon, t_max_lon = self.wind_turbines_df['lon'].min(), self.wind_turbines_df['lon'].max()
                    t_min_lat, t_max_lat = self.wind_turbines_df['lat'].min(), self.wind_turbines_df['lat'].max()
                    buffer = 0.01
                    for _ in range(remaining):
                        nest_lons.append(np.random.uniform(t_min_lon - buffer, t_max_lon + buffer))
                        nest_lats.append(np.random.uniform(t_min_lat - buffer, t_max_lat + buffer))

        nest_coords = [(lon, lat) for lon, lat in zip(nest_lons, nest_lats)]
        nest_z_ground = []
        for val in self.dem_src.sample(nest_coords):
            z = val[0]
            if z < -1000 or z == self.nodata_val: z = self.min_elevation_dem
            nest_z_ground.append(z)
        
        self.nest_locations = pd.DataFrame({'id': range(len(nest_lons)), 'lon': nest_lons, 'lat': nest_lats, 'z_ground': nest_z_ground})

    def load_wind_turbines(self):
        if not os.path.exists(config.OSM_FILE): return pd.DataFrame(columns=['id', 'lat', 'lon', 'z_ground'])
        df = pd.read_csv(config.OSM_FILE)
        mask = (df['lon'] >= self.bounds.left) & (df['lon'] <= self.bounds.right) & \
               (df['lat'] >= self.bounds.bottom) & (df['lat'] <= self.bounds.top)
        df = df[mask].copy()
        if df.empty: return df
        tz = []
        for val in self.dem_src.sample(zip(df['lon'], df['lat'])):
            z = val[0]
            if z < -1000: z = self.min_elevation_dem
            tz.append(z)
        df['z_ground'] = tz
        return df

    def calculate_max_horizontal_distance_limit(self, start_lon, start_lat, start_z, direction_theta):
        """
        【修正版 - 动态地速/能量模型】
        """
        cache_key = (start_lon, start_lat, start_z, direction_theta)
        if cache_key in self._max_distance_limit_cache:
            return self._max_distance_limit_cache[cache_key]

        # 1. 计算去程和回程的动态参数 (地速 Vg 和 功率 P)
        Vg_out, P_out = self.calculate_dynamic_flight_parameters(direction_theta)
        Vg_in, P_in   = self.calculate_dynamic_flight_parameters(direction_theta + 180)
        
        # 2. 计算水平总能耗率 (J/m)
        # 去程每米耗能 = P_out / Vg_out
        # 回程每米耗能 = P_in / Vg_in
        energy_rate_out = P_out / Vg_out
        energy_rate_in  = P_in / Vg_in
        total_horiz_energy_rate = energy_rate_out + energy_rate_in
        
        # 3. 搜索范围
        max_theoretical_dist = config.E_BAT / total_horiz_energy_rate
        step_size = 30.0 
        num_steps = int(max_theoretical_dist / step_size)
        if num_steps <= 0: return 0.0

        dist_array = np.linspace(0, max_theoretical_dist, num_steps)

        # 4. 路径采样
        theta_rad = np.radians(direction_theta)
        mean_lat_rad = np.radians(start_lat)
        m_per_deg_lat = 111000.0
        m_per_deg_lon = 111000.0 * np.cos(mean_lat_rad)
        
        d_lat_vec = (dist_array * np.cos(theta_rad)) / m_per_deg_lat
        d_lon_vec = (dist_array * np.sin(theta_rad)) / m_per_deg_lon
        
        path_lats = start_lat + d_lat_vec
        path_lons = start_lon + d_lon_vec
        
        rows, cols = rasterio.transform.rowcol(self.transform, path_lons, path_lats)
        rows = np.array(rows); cols = np.array(cols)
        
        valid_mask = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
        if not valid_mask[0]: 
            self._max_distance_limit_cache[cache_key] = 0.0
            return 0.0
            
        first_invalid = np.where(~valid_mask)[0]
        if len(first_invalid) > 0:
            limit_idx = first_invalid[0]
            rows = rows[:limit_idx]; cols = cols[:limit_idx]; dist_array = dist_array[:limit_idx]
            
        path_elevations = self.elevation_full[rows, cols]
        path_elevations = np.nan_to_num(path_elevations, nan=self.min_elevation_dem)
        
        path_max_elevs = np.maximum.accumulate(path_elevations)
        
        # 5. 计算垂直能耗
        safe_h = np.maximum(start_z, path_elevations)
        safe_h = np.maximum(safe_h, path_max_elevs)
        safe_h += config.DELTA_SAFE
        
        t_climb_out = np.maximum(0, safe_h - start_z) / config.V_UP
        t_desc_out  = np.maximum(0, safe_h - path_elevations) / config.V_DOWN
        t_climb_in = np.maximum(0, safe_h - path_elevations) / config.V_UP
        t_desc_in  = np.maximum(0, safe_h - start_z) / config.V_DOWN
        
        E_vert_out = self.P_climb_approx * t_climb_out + self.P_desc_approx * t_desc_out
        E_vert_in  = self.P_climb_approx * t_climb_in + self.P_desc_approx * t_desc_in
        
        E_serv = self.P_hover * config.T_SERV
        
        # 6. 总能耗
        E_horiz_total = total_horiz_energy_rate * dist_array
        E_total_mission = E_horiz_total + E_vert_out + E_vert_in + E_serv
        
        feasible_indices = np.where(E_total_mission <= config.E_BAT)[0]
        
        if len(feasible_indices) == 0:
            result = 0.0
        else:
            result = dist_array[feasible_indices[-1]]
            
        self._max_distance_limit_cache[cache_key] = result
        return result

    def calculate_reachability_matrix(self, nest_locations_df, turbines_df):
        n_nests = len(nest_locations_df)
        n_turbines = len(turbines_df)
        R_lk = np.zeros((n_nests, n_turbines), dtype=int)
        
        print(f"Calculating R_lk (Dynamic Speed) for {n_nests} nests and {n_turbines} turbines...")
        
        for idx_i, (nest_idx, nest_row) in enumerate(nest_locations_df.iterrows()):
            n_lon, n_lat, n_z = nest_row['lon'], nest_row['lat'], nest_row['z_ground']
            if idx_i % 10 == 0: print(f"Processing Nest {idx_i}...")

            for idx_j, (turb_idx, turb_row) in enumerate(turbines_df.iterrows()):
                t_lon, t_lat = turb_row['lon'], turb_row['lat']
                
                mean_lat = np.radians((n_lat + t_lat)/2)
                dx = (t_lon - n_lon) * 111000 * np.cos(mean_lat)
                dy = (t_lat - n_lat) * 111000
                dist_m = np.sqrt(dx**2 + dy**2)
                
                angle_rad = np.arctan2(dx, dy) 
                angle_deg = np.degrees(angle_rad) % 360
                
                d_limit = self.calculate_max_horizontal_distance_limit(n_lon, n_lat, n_z, angle_deg)
                
                if dist_m <= d_limit:
                    R_lk[idx_i, idx_j] = 1
                else:
                    R_lk[idx_i, idx_j] = 0
        
        print(f"R_lk calculated. Reachable pairs: {np.sum(R_lk)} / {R_lk.size}")
        return R_lk