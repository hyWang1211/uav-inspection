# phase1_optimizer.py
# Version 4: Lock-in Strategy (Guarantees Feasibility)

import pandas as pd
import numpy as np
import math
from Model import config  # 导入全局配置

class Phase1Optimizer:
    def __init__(self, physical_model):
        self.physical_model = physical_model
        self.nest_locations_df = physical_model.nest_locations
        self.turbines_df = physical_model.wind_turbines_df

        if self.nest_locations_df.empty or self.turbines_df.empty:
            raise ValueError("Nest locations or wind turbine data is missing.")

        self.n_nests = len(self.nest_locations_df)
        self.n_turbines = len(self.turbines_df)
        self.nest_ids = list(range(self.n_nests))
        self.turbine_ids = list(range(self.n_turbines))
        
        # --- 核心约束参数 ---
        self.eta = config.ETA  # 单机服务上限 (例如 4)
        self.base_costs = config.NEST_BASE_COSTS
        self.uav_cost = config.OMEGA_2
        self.max_type = max(self.base_costs.keys()) 
        
        # 单个机巢绝对最大容量 (例如 4 * 4 = 16)
        self.global_capacity_limit = self.max_type * self.eta
        
        # 预先计算 R_lk 矩阵
        print(f"\n--- Phase I (Heuristic V4): Max Capacity per Nest = {self.global_capacity_limit} ---")
        self.R_lk = self.physical_model.calculate_reachability_matrix(
            self.nest_locations_df, 
            self.turbines_df
        )

        # 预计算距离矩阵
        self.dist_matrix = self._calculate_distance_matrix()

    def _calculate_distance_matrix(self):
        dists = np.full((self.n_nests, self.n_turbines), 1e9)
        nest_coords = self.nest_locations_df[['lon', 'lat']].values
        turb_coords = self.turbines_df[['lon', 'lat']].values
        
        for l in range(self.n_nests):
            for k in range(self.n_turbines):
                if self.R_lk[l, k] == 1:
                    dx = nest_coords[l][0] - turb_coords[k][0]
                    dy = nest_coords[l][1] - turb_coords[k][1]
                    dists[l, k] = np.sqrt(dx**2 + dy**2)
        return dists

    def _get_required_type_and_cost(self, num_turbines):
        """根据风机数量计算机巢类型和成本"""
        if num_turbines == 0: return 0, 0
        n_uavs = math.ceil(num_turbines / self.eta)
        nest_type = n_uavs
        if nest_type > self.max_type: nest_type = self.max_type
        if nest_type < 1: nest_type = 1
        cost = self.base_costs[nest_type] + nest_type * self.uav_cost
        return nest_type, cost

    def _greedy_construction_locked(self):
        """
        修正后的贪婪选址：选定即锁定 (Lock-in)
        返回：
            assignment: dict {turbine_id: nest_id}
            active_nests: set {nest_id}
            nest_loads: dict {nest_id: count}
        """
        uncovered_turbines = set(self.turbine_ids)
        
        # 核心改变：直接在构建过程中维护最终状态
        final_assignment = {} 
        nest_loads = {}       # 记录每个已选机巢的当前负载
        
        print("Starting Locked Greedy Construction...")
        
        while uncovered_turbines:
            best_score = float('inf')
            best_nest_id = -1
            best_turbines_to_take = []

            # 遍历所有候选机巢 (尚未激活的)
            candidate_nests = [l for l in self.nest_ids if l not in nest_loads]
            
            # 如果没有新机巢可选了，但还有风机没覆盖 -> 无解 (通常是物理不可达)
            if not candidate_nests:
                print(f"CRITICAL WARNING: Run out of candidates! {len(uncovered_turbines)} turbines remain uncovered.")
                # 强行把剩下的标记为 -1
                for k in uncovered_turbines:
                    final_assignment[k] = -1
                break

            for l in candidate_nests:
                # 1. 找出该机巢能到达的、且目前未覆盖的风机
                reachable_indices = np.where(self.R_lk[l] == 1)[0]
                potential_turbines = [k for k in reachable_indices if k in uncovered_turbines]
                
                if not potential_turbines:
                    continue

                # 2. 截断：最多只能吃 global_capacity_limit 个
                # 按距离排序，优先吃掉最近的
                potential_turbines.sort(key=lambda k: self.dist_matrix[l, k])
                real_turbines = potential_turbines[:self.global_capacity_limit]
                gain = len(real_turbines)
                
                # 3. 计算性价比
                _, cost = self._get_required_type_and_cost(gain)
                score = cost / gain
                
                if score < best_score:
                    best_score = score
                    best_nest_id = l
                    best_turbines_to_take = real_turbines

            if best_nest_id != -1:
                # 选中最佳机巢，立即锁定分配
                nest_loads[best_nest_id] = len(best_turbines_to_take)
                for k in best_turbines_to_take:
                    final_assignment[k] = best_nest_id
                    uncovered_turbines.remove(k)
            else:
                print(f"Warning: {len(uncovered_turbines)} turbines are physically unreachable.")
                break
        
        return final_assignment, list(nest_loads.keys()), nest_loads

    def _optimize_and_prune_locked(self, initial_assignment, active_nests, initial_loads):
        """
        基于锁定的剪枝：
        只有当一个机巢的所有任务都能被【其他已有机巢的剩余容量】接管时，才移除它。
        """
        print("Starting Pruning (Capacity-Aware)...")
        
        current_assignment = initial_assignment.copy()
        current_active = set(active_nests)
        current_loads = initial_loads.copy()
        
        improved = True
        while improved:
            improved = False
            # 优先尝试移除负载小的机巢
            sorted_nests = sorted(list(current_active), key=lambda l: current_loads[l])
            
            for l_remove in sorted_nests:
                # 找出它负责的所有风机
                my_turbines = [k for k, v in current_assignment.items() if v == l_remove]
                
                # 模拟移除：尝试为 my_turbines 里的每一个找到新家
                # 新家必须在 current_active 中，且不是 l_remove，且有剩余容量
                
                can_reassign_all = True
                moves = {} # turbine -> new_nest
                
                # 创建临时负载表用于检查
                temp_loads = current_loads.copy()
                del temp_loads[l_remove]
                
                cost_delta = 0
                # 节省的成本
                _, saved = self._get_required_type_and_cost(current_loads[l_remove])
                cost_delta -= saved
                
                for k in my_turbines:
                    # 找可用的邻居
                    candidates = []
                    for l_target in current_active:
                        if l_target == l_remove: continue
                        if self.R_lk[l_target, k] == 1:
                            if temp_loads[l_target] < self.global_capacity_limit:
                                candidates.append(l_target)
                    
                    if not candidates:
                        can_reassign_all = False
                        break
                    
                    # 选距离最近的
                    candidates.sort(key=lambda l: self.dist_matrix[l, k])
                    best_new_home = candidates[0]
                    
                    # 记录移动
                    moves[k] = best_new_home
                    
                    # 计算增加的成本
                    old_c = self._get_required_type_and_cost(temp_loads[best_new_home])[1]
                    new_c = self._get_required_type_and_cost(temp_loads[best_new_home] + 1)[1]
                    cost_delta += (new_c - old_c)
                    
                    # 更新临时负载
                    temp_loads[best_new_home] += 1
                
                if can_reassign_all and cost_delta < 0:
                    print(f"  Pruning nest {l_remove}, saving {-cost_delta:.2f}")
                    
                    # 执行移动
                    current_active.remove(l_remove)
                    current_loads = temp_loads # 更新负载表
                    for k, new_home in moves.items():
                        current_assignment[k] = new_home
                    
                    improved = True
                    break # 重新开始循环
                    
        return current_assignment, list(current_active), current_loads

    def run(self):
        print("\n--- Running Heuristic Optimizer (Locked V4) ---")
        
        # 1. 锁定式贪婪构建 (保证初始可行性)
        assignment, active_nests, loads = self._greedy_construction_locked()
        print(f"Greedy construction selected {len(active_nests)} nests.")
        
        # 检查是否有未分配的
        unassigned = [k for k, v in assignment.items() if v == -1]
        if unassigned:
            print(f"CRITICAL: {len(unassigned)} turbines are unreachable by any nest!")
        
        # 2. 安全剪枝
        final_assignment, final_nests, final_loads = self._optimize_and_prune_locked(assignment, active_nests, loads)
        print(f"After pruning, {len(final_nests)} nests remain.")
        
        # 3. 输出格式化
        selected_nests_data = []
        total_cost = 0
        
        for l in final_nests:
            n = final_loads[l]
            nest_type, cost = self._get_required_type_and_cost(n)
            selected_nests_data.append({
                'nest_id': l,
                'capacity': nest_type,
                'load': n
            })
            total_cost += cost
            
        assignments_data = []
        for k, l in final_assignment.items():
            if l != -1:
                assignments_data.append({'turbine_id': k, 'nest_id': l})
        
        print(f"Final Total Cost: {total_cost:.2f}")
        return pd.DataFrame(selected_nests_data), pd.DataFrame(assignments_data), total_cost